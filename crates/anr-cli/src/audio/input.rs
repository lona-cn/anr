use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::Path,
    sync::{Arc, Mutex, atomic::AtomicBool},
};

use anyhow::{Context, Result, bail, ensure};
use symphonia::core::{
    audio::{SampleBuffer, SignalSpec},
    codecs::{CodecParameters, Decoder, DecoderOptions},
    errors::Error as DecodeError,
    formats::{FormatOptions, FormatReader},
    io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions, ReadBytes, SeekBuffered},
    meta::MetadataOptions,
    probe::{Instantiate, QueryDescriptor},
};

use super::{AudioInfo, AudioSource, AudioSpec, stdio::CancellableReader};
use crate::{args::InputFormat, cancel};

const CHUNK_FRAMES: usize = 4096;

pub fn open_input(
    path: &Path,
    format: InputFormat,
    raw_spec: Option<AudioSpec>,
    cancelled: Arc<AtomicBool>,
) -> Result<Box<dyn AudioSource>> {
    cancel::check(&cancelled)?;
    let stdin = path == Path::new("-");
    match format {
        InputFormat::F32le => {
            let spec =
                raw_spec.context("raw f32le input requires a sample rate and channel count")?;
            let (reader, bytes): (Box<dyn Read>, Option<u64>) = if stdin {
                (
                    Box::new(CancellableReader::spawn(
                        std::io::stdin(),
                        cancelled.clone(),
                    )?),
                    None,
                )
            } else {
                let file = File::open(path)
                    .with_context(|| format!("opening input {}", path.display()))?;
                let bytes = file.metadata()?.len();
                (Box::new(file), Some(bytes))
            };
            Ok(Box::new(RawSource::new(reader, spec, bytes, cancelled)?))
        }
        InputFormat::Auto => {
            ensure!(
                raw_spec.is_none(),
                "raw specifications cannot override encoded input"
            );
            let source: Box<dyn MediaSource> = if stdin {
                Box::new(NonSeekSource(Mutex::new(CancellableReader::spawn(
                    std::io::stdin(),
                    cancelled.clone(),
                )?)))
            } else {
                Box::new(
                    File::open(path)
                        .with_context(|| format!("opening input {}", path.display()))?,
                )
            };
            Ok(Box::new(EncodedSource::open(source, cancelled)?))
        }
    }
}

fn io_error(error: io::Error) -> anyhow::Error {
    if error
        .get_ref()
        .is_some_and(|inner| inner.is::<cancel::Cancelled>())
    {
        cancel::Cancelled.into()
    } else {
        error.into()
    }
}

fn decode_error(error: DecodeError) -> anyhow::Error {
    // Symphonia 0.5 exposes the nested IO error only through deprecated cause(),
    // not source(). Unwrap it so callers retain BrokenPipe/Cancelled identities.
    match error {
        DecodeError::IoError(error) => io_error(error),
        other => other.into(),
    }
}

struct NonSeekSource(Mutex<CancellableReader>);

impl Read for NonSeekSource {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        self.0
            .get_mut()
            .map_err(|_| io::Error::other("stdin reader lock poisoned"))?
            .read(output)
    }
}
impl Seek for NonSeekSource {
    fn seek(&mut self, _: SeekFrom) -> io::Result<u64> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "encoded stdin is not seekable; use raw f32le for streaming",
        ))
    }
}
impl MediaSource for NonSeekSource {
    fn is_seekable(&self) -> bool {
        false
    }
    fn byte_len(&self) -> Option<u64> {
        None
    }
}

struct RawSource {
    reader: Box<dyn Read>,
    info: AudioInfo,
    bytes: Vec<u8>,
    frame_bytes: usize,
    eof: bool,
    cancelled: Arc<AtomicBool>,
}

impl RawSource {
    fn new(
        reader: Box<dyn Read>,
        spec: AudioSpec,
        bytes: Option<u64>,
        cancelled: Arc<AtomicBool>,
    ) -> Result<Self> {
        ensure!(
            spec.sample_rate > 0 && spec.channels > 0,
            "invalid raw audio specification"
        );
        let frame_bytes = spec
            .channels
            .checked_mul(4)
            .context("raw channel count is too large")?;
        let capacity = frame_bytes
            .checked_mul(CHUNK_FRAMES)
            .context("raw buffer is too large")?;
        let frames = bytes
            .filter(|n| n % frame_bytes as u64 == 0)
            .map(|n| n / frame_bytes as u64);
        Ok(Self {
            reader,
            info: AudioInfo {
                format: "f32le".into(),
                codec: "pcm_f32le".into(),
                track_id: None,
                spec,
                frames,
                duration_seconds: frames.map(|n| n as f64 / spec.sample_rate as f64),
            },
            bytes: vec![0; capacity],
            frame_bytes,
            eof: false,
            cancelled,
        })
    }
}

impl AudioSource for RawSource {
    fn info(&self) -> &AudioInfo {
        &self.info
    }

    fn read_frames(&mut self, output: &mut [Vec<f32>]) -> Result<usize> {
        cancel::check(&self.cancelled)?;
        ensure!(
            output.len() == self.info.spec.channels,
            "input channel buffer count does not match"
        );
        for channel in output.iter_mut() {
            channel.clear();
        }
        if self.eof {
            return Ok(0);
        }
        let mut valid = 0;
        while valid < self.bytes.len() {
            cancel::check(&self.cancelled)?;
            match self.reader.read(&mut self.bytes[valid..]) {
                Ok(0) => {
                    self.eof = true;
                    break;
                }
                Ok(count) => valid += count,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(io_error(error)).context("reading raw f32le input"),
            }
        }
        ensure!(
            valid % self.frame_bytes == 0,
            "truncated raw f32le input: EOF inside an audio frame"
        );
        let frames = valid / self.frame_bytes;
        for frame in self.bytes[..valid].chunks_exact(self.frame_bytes) {
            for (channel, bytes) in output.iter_mut().zip(frame.as_chunks::<4>().0) {
                let value = f32::from_le_bytes(*bytes);
                ensure!(value.is_finite(), "non-finite raw input audio sample");
                channel.push(value);
            }
        }
        Ok(frames)
    }
}

// ProbeResult omits the selected container name. Use the exact marker at the
// position selected by Symphonia, matched against its enabled descriptors (not
// an extension, which may be wrong). Metadata such as ID3 is consumed by its
// registered reader before continuing the same probe.
fn probe_format(
    source: Box<dyn MediaSource>,
    cancelled: &AtomicBool,
) -> Result<(String, Box<dyn FormatReader>)> {
    use symphonia::default::formats::*;
    let mut descriptors = Vec::new();
    descriptors.extend_from_slice(AdtsReader::query());
    descriptors.extend_from_slice(CafReader::query());
    descriptors.extend_from_slice(FlacReader::query());
    descriptors.extend_from_slice(IsoMp4Reader::query());
    descriptors.extend_from_slice(MpaReader::query());
    descriptors.extend_from_slice(AiffReader::query());
    descriptors.extend_from_slice(WavReader::query());
    descriptors.extend_from_slice(OggReader::query());
    descriptors.extend_from_slice(MkvReader::query());
    let mut stream = MediaSourceStream::new(source, MediaSourceStreamOptions::default());
    loop {
        cancel::check(cancelled)?;
        let result = symphonia::default::get_probe().next(&mut stream);
        cancel::check(cancelled)?;
        match result
            .map_err(decode_error)
            .context("probing encoded input")?
        {
            Instantiate::Metadata(create) => {
                create(&MetadataOptions::default())
                    .read_all(&mut stream)
                    .map_err(decode_error)?;
            }
            Instantiate::Format(create) => {
                let mut marker = [0; 16];
                ReadBytes::read_buf_exact(&mut stream, &mut marker).map_err(io_error)?;
                stream.seek_buffered_rev(marker.len());
                let name = descriptors
                    .iter()
                    .find(|descriptor| {
                        descriptor
                            .markers
                            .iter()
                            .any(|prefix| marker.starts_with(prefix))
                    })
                    .context("probed container has no registered descriptor")?
                    .short_name
                    .to_owned();
                let result = create(
                    stream,
                    &FormatOptions {
                        enable_gapless: true,
                        ..Default::default()
                    },
                );
                cancel::check(cancelled)?;
                return Ok((
                    name,
                    result
                        .map_err(decode_error)
                        .context("opening encoded container")?,
                ));
            }
        }
    }
}

struct EncodedSource {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    info: AudioInfo,
    signal: Option<SignalSpec>,
    samples: Option<SampleBuffer<f32>>,
    sample_offset: usize,
    cancelled: Arc<AtomicBool>,
}

impl EncodedSource {
    fn open(source: Box<dyn MediaSource>, cancelled: Arc<AtomicBool>) -> Result<Self> {
        let (format_name, format) = probe_format(source, &cancelled)?;
        let mut selected = None;
        let mut last_error = None;
        for track in format.tracks() {
            match symphonia::default::get_codecs()
                .make(&track.codec_params, &DecoderOptions::default())
            {
                Ok(decoder) => {
                    selected = Some((track.id, track.codec_params.clone(), decoder));
                    break;
                }
                Err(error) => last_error = Some(decode_error(error)),
            }
        }
        let (track_id, parameters, decoder) = match selected {
            Some(selected) => selected,
            None => {
                return Err(last_error
                    .unwrap_or_else(|| anyhow::anyhow!("container contains no audio tracks")))
                .context("no decodable audio track");
            }
        };
        let codec = symphonia::default::get_codecs()
            .get_codec(parameters.codec)
            .context("selected codec is not registered")?
            .short_name
            .to_owned();
        let spec = AudioSpec {
            sample_rate: parameters.sample_rate.unwrap_or(0),
            channels: parameters.channels.map_or(0, |channels| channels.count()),
        };
        let signal = parameters
            .sample_rate
            .zip(parameters.channels)
            .map(|(rate, channels)| SignalSpec::new(rate, channels));
        let mut this = Self {
            format,
            decoder,
            info: AudioInfo {
                format: format_name,
                codec,
                track_id: Some(track_id),
                spec,
                frames: None,
                duration_seconds: None,
            },
            signal,
            samples: None,
            sample_offset: 0,
            cancelled,
        };
        if spec.sample_rate == 0 || spec.channels == 0 {
            ensure!(
                this.decode_packet()?,
                "audio stream ended before its specification was available"
            );
        }
        ensure!(
            this.info.spec.sample_rate > 0 && this.info.spec.channels > 0,
            "invalid decoded audio specification"
        );
        this.set_duration(&parameters);
        Ok(this)
    }

    fn set_duration(&mut self, parameters: &CodecParameters) {
        if let Some(ticks) = parameters.n_frames {
            if let Some(base) = parameters.time_base {
                let time = base.calc_time(ticks);
                self.info.duration_seconds = Some(time.seconds as f64 + time.frac);
                let numerator = u128::from(ticks)
                    * u128::from(base.numer)
                    * u128::from(self.info.spec.sample_rate);
                self.info.frames = u64::try_from(numerator / u128::from(base.denom)).ok();
            } else {
                self.info.frames = Some(ticks);
                self.info.duration_seconds =
                    Some(ticks as f64 / f64::from(self.info.spec.sample_rate));
            }
        }
    }

    fn decode_packet(&mut self) -> Result<bool> {
        loop {
            cancel::check(&self.cancelled)?;
            let result = self.format.next_packet();
            // Streaming containers can append tags while finding the next
            // packet (or EOF). We do not expose tags, so retain only the newest
            // revision rather than letting old cover art and values accumulate.
            self.format.metadata().skip_to_latest();
            cancel::check(&self.cancelled)?;
            let packet = match result {
                Ok(packet) => packet,
                Err(DecodeError::IoError(error))
                    if error.kind() == io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(false);
                }
                Err(DecodeError::ResetRequired) => {
                    bail!("encoded input changed tracks or audio specification")
                }
                Err(error) => {
                    return Err(decode_error(error)).context("reading encoded audio packet");
                }
            };
            if Some(packet.track_id()) != self.info.track_id {
                continue;
            }
            let decoded = self
                .decoder
                .decode(&packet)
                .map_err(decode_error)
                .context("decoding audio packet")?;
            let signal = *decoded.spec();
            ensure!(
                signal.rate > 0 && signal.channels.count() > 0,
                "invalid decoded audio specification"
            );
            if let Some(previous) = self.signal {
                ensure!(previous == signal, "audio specification changed midstream");
            } else {
                self.signal = Some(signal);
                self.info.spec = AudioSpec {
                    sample_rate: signal.rate,
                    channels: signal.channels.count(),
                };
            }
            let needed = decoded
                .capacity()
                .checked_mul(signal.channels.count())
                .context("decoded packet is too large")?;
            if self
                .samples
                .as_ref()
                .is_none_or(|buffer| buffer.capacity() < needed)
            {
                self.samples = Some(SampleBuffer::new(decoded.capacity() as u64, signal));
            }
            let samples = self.samples.as_mut().expect("sample buffer allocated");
            samples.copy_interleaved_ref(decoded);
            self.sample_offset = 0;
            ensure!(
                samples.samples().iter().all(|value| value.is_finite()),
                "non-finite decoded input audio sample"
            );
            if !samples.is_empty() {
                return Ok(true);
            }
        }
    }
}

impl AudioSource for EncodedSource {
    fn info(&self) -> &AudioInfo {
        &self.info
    }

    fn read_frames(&mut self, output: &mut [Vec<f32>]) -> Result<usize> {
        cancel::check(&self.cancelled)?;
        ensure!(
            output.len() == self.info.spec.channels,
            "input channel buffer count does not match"
        );
        for channel in output.iter_mut() {
            channel.clear();
        }
        if self
            .samples
            .as_ref()
            .is_none_or(|samples| self.sample_offset == samples.len())
            && !self.decode_packet()?
        {
            return Ok(0);
        }
        let samples = self
            .samples
            .as_ref()
            .context("decoder returned no audio buffer")?
            .samples();
        let channels = self.info.spec.channels;
        let frames = ((samples.len() - self.sample_offset) / channels).min(CHUNK_FRAMES);
        let end = self.sample_offset + frames * channels;
        for frame in samples[self.sample_offset..end].chunks_exact(channels) {
            for (channel, &sample) in output.iter_mut().zip(frame) {
                channel.push(sample);
            }
        }
        self.sample_offset = end;
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    struct Fragmented {
        inner: Cursor<Vec<u8>>,
        next: usize,
    }
    impl Read for Fragmented {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let n = out.len().min(self.next);
            self.next = self.next % 3 + 1;
            self.inner.read(&mut out[..n])
        }
    }
    fn flag() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }
    fn raw(bytes: Vec<u8>, channels: usize) -> RawSource {
        RawSource::new(
            Box::new(Fragmented {
                inner: Cursor::new(bytes),
                next: 1,
            }),
            AudioSpec {
                sample_rate: 48000,
                channels,
            },
            None,
            flag(),
        )
        .unwrap()
    }
    #[test]
    fn short_reads_reassemble_stereo_frames_without_reordering() {
        let values = [0.25_f32, -0.5, 1.0, -1.0, 0.0, 0.75];
        let mut source = raw(values.into_iter().flat_map(f32::to_le_bytes).collect(), 2);
        let mut out = vec![Vec::new(), Vec::new()];
        let mut all = [Vec::new(), Vec::new()];
        loop {
            let n = source.read_frames(&mut out).unwrap();
            if n == 0 {
                break;
            }
            for c in 0..2 {
                all[c].extend_from_slice(&out[c]);
            }
        }
        assert_eq!(all[0], [0.25, 1.0, 0.0]);
        assert_eq!(all[1], [-0.5, -1.0, 0.75]);
    }
    #[test]
    fn eof_rejects_partial_samples_and_partial_channel_frames() {
        for bytes in [vec![0, 0, 0], 1.0_f32.to_le_bytes().to_vec()] {
            let mut source = raw(bytes, 2);
            assert!(source.read_frames(&mut [Vec::new(), Vec::new()]).is_err());
        }
    }
    #[test]
    fn raw_rejects_nonfinite_audio() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut source = raw(value.to_le_bytes().to_vec(), 1);
            assert!(source.read_frames(&mut [Vec::new()]).is_err());
        }
    }
    #[test]
    fn raw_reads_are_bounded_and_preserve_last_frame() {
        let mut source = raw(vec![0; (CHUNK_FRAMES + 3) * 4], 1);
        let mut out = [Vec::new()];
        let mut total = 0;
        loop {
            let n = source.read_frames(&mut out).unwrap();
            assert!(n <= CHUNK_FRAMES);
            if n == 0 {
                break;
            }
            total += n;
        }
        assert_eq!(total, CHUNK_FRAMES + 3);
    }
    fn wav_bytes(frames: usize, nonfinite: bool) -> Vec<u8> {
        let mut bytes = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(
                &mut bytes,
                hound::WavSpec {
                    channels: 2,
                    sample_rate: 44100,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            for frame in 0..frames {
                writer
                    .write_sample(if nonfinite {
                        f32::NAN
                    } else {
                        frame as f32 / 10000.0
                    })
                    .unwrap();
                writer.write_sample(-0.5_f32).unwrap();
            }
            writer.finalize().unwrap();
        }
        bytes.into_inner()
    }

    #[test]
    fn encoded_stereo_preserves_samples_and_uses_content_not_extension() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("actually-wave.mp3");
        std::fs::write(&path, wav_bytes(CHUNK_FRAMES + 3, false)).unwrap();
        let mut source = open_input(&path, InputFormat::Auto, None, flag()).unwrap();
        assert_eq!(source.info().format, "wave");
        assert_eq!(
            source.info().spec,
            AudioSpec {
                sample_rate: 44100,
                channels: 2
            }
        );
        assert_eq!(source.info().frames, Some((CHUNK_FRAMES + 3) as u64));
        let mut out = [Vec::new(), Vec::new()];
        let mut total = 0;
        loop {
            let frames = source.read_frames(&mut out).unwrap();
            assert!(frames <= CHUNK_FRAMES);
            assert_eq!(out[0].len(), frames);
            assert_eq!(out[1].len(), frames);
            if frames == 0 {
                break;
            }
            for (frame, (left, right)) in out[0].iter().zip(&out[1]).enumerate() {
                assert_eq!(*left, (total + frame) as f32 / 10000.0);
                assert_eq!(*right, -0.5);
            }
            total += frames;
        }
        assert_eq!(total, CHUNK_FRAMES + 3);
    }

    #[test]
    fn encoded_wave_can_read_nonseekable_stream() {
        let source = NonSeekSource(Mutex::new(
            CancellableReader::spawn(
                Fragmented {
                    inner: Cursor::new(wav_bytes(3, false)),
                    next: 1,
                },
                flag(),
            )
            .unwrap(),
        ));
        let mut source = EncodedSource::open(Box::new(source), flag()).unwrap();
        let mut output = [Vec::new(), Vec::new()];
        assert_eq!(source.read_frames(&mut output).unwrap(), 3);
        assert_eq!(output[1], [-0.5; 3]);
        assert_eq!(source.read_frames(&mut output).unwrap(), 0);
    }

    #[test]
    fn encoded_float_rejects_nonfinite_samples() {
        let mut source =
            EncodedSource::open(Box::new(Cursor::new(wav_bytes(1, true))), flag()).unwrap();
        assert!(source.read_frames(&mut [Vec::new(), Vec::new()]).is_err());
    }
    #[test]
    fn repeated_matroska_tags_do_not_accumulate_during_audio_decode() {
        fn element(id: &[u8], payload: &[u8]) -> Vec<u8> {
            assert!(payload.len() < 0x3fff);
            let mut bytes = id.to_vec();
            if payload.len() < 0x7f {
                bytes.push(0x80 | payload.len() as u8);
            } else {
                bytes.extend_from_slice(&(0x4000 | payload.len() as u16).to_be_bytes());
            }
            bytes.extend_from_slice(payload);
            bytes
        }
        let ebml = element(
            &[0x1a, 0x45, 0xdf, 0xa3],
            &element(&[0x42, 0x82], b"matroska"),
        );
        let mut audio = element(&[0xb5], &48000_f64.to_be_bytes());
        audio.extend(element(&[0x9f], &[1]));
        let mut track = element(&[0xd7], &[1]);
        track.extend(element(&[0x73, 0xc5], &[1]));
        track.extend(element(&[0x83], &[2]));
        track.extend(element(&[0x86], b"A_MPEG/L3"));
        track.extend(element(&[0x23, 0xe3, 0x83], &24_000_000_u32.to_be_bytes()));
        track.extend(element(&[0xe1], &audio));
        // Symphonia 0.5 treats a zero-length master as unknown-sized, so Info
        // and Tags need real children instead of empty master placeholders.
        let mut segment = element(
            &[0x15, 0x49, 0xa9, 0x66],
            &element(&[0x2a, 0xd7, 0xb1], &[0x0f, 0x42, 0x40]),
        );
        segment.extend(element(
            &[0x16, 0x54, 0xae, 0x6b],
            &element(&[0xae], &track),
        ));
        let mut simple_tag = element(&[0x45, 0xa3], b"TITLE");
        simple_tag.extend(element(&[0x44, 0x87], b"stream metadata revision"));
        let tags = element(
            &[0x12, 0x54, 0xc3, 0x67],
            &element(&[0x73, 0x73], &element(&[0x67, 0xc8], &simple_tag)),
        );
        // Self-authored MPEG-1 Layer III silence: 128 kb/s, 48 kHz, mono,
        // no CRC or padding. 144 * 128000 / 48000 = 384 bytes per frame.
        // The 17-byte side information and remaining main data are zero:
        // main_data_begin, scalefac_compress, part2_3_length and big_values
        // are all zero, so both 576-sample granules decode as silence.
        let mut mpeg_frame = [0; 384];
        mpeg_frame[..4].copy_from_slice(&[0xff, 0xfb, 0x94, 0xc0]);
        for frame in 0..16_u8 {
            // Tags are segment-level siblings, not cluster metadata. Multiple
            // revisions between packets must be retired, including at EOF.
            for _ in 0..3 {
                segment.extend_from_slice(&tags);
            }
            let mut cluster = element(&[0xe7], &(u16::from(frame) * 24).to_be_bytes());
            let mut block = vec![0x81, 0, 0, 0x80];
            block.extend_from_slice(&mpeg_frame);
            cluster.extend(element(&[0xa3], &block));
            segment.extend(element(&[0x1f, 0x43, 0xb6, 0x75], &cluster));
        }
        for _ in 0..3 {
            segment.extend_from_slice(&tags);
        }
        let mut bytes = ebml;
        bytes.extend(element(&[0x18, 0x53, 0x80, 0x67], &segment));
        let mut source = EncodedSource::open(Box::new(Cursor::new(bytes)), flag()).unwrap();
        let mut output = [Vec::new()];
        let mut frames = 0;
        loop {
            let count = source.read_frames(&mut output).unwrap();
            assert!(
                source.format.metadata().is_latest(),
                "old tag revisions remain retained"
            );
            assert_eq!(source.format.metadata().current().unwrap().tags().len(), 1);
            if count == 0 {
                break;
            }
            assert_eq!(count, 1152);
            assert!(output[0].iter().all(|&sample| sample == 0.0));
            frames += count;
        }
        assert_eq!(frames, 16 * 1152);
    }
}
