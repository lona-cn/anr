use anyhow::{Context, Result, bail, ensure};
use hound::{SampleFormat, WavReader};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, Seek};
use std::path::{Path, PathBuf};

pub(super) struct Pair {
    pub id: String,
    pub speaker: String,
    pub noisy: PathBuf,
    pub clean: PathBuf,
    pub samples: usize,
}

pub(super) struct Corpus {
    pub train: Vec<Pair>,
    pub validation: Vec<Pair>,
    pub test: Vec<Pair>,
    pub validation_speakers: Vec<String>,
}

pub(super) fn discover(root: &Path, validation_speakers: &[String]) -> Result<Corpus> {
    let training = discover_pairs(
        &root.join("noisy_trainset_28spk_wav"),
        &root.join("clean_trainset_28spk_wav"),
    )?;
    let test = discover_pairs(
        &root.join("noisy_testset_wav"),
        &root.join("clean_testset_wav"),
    )?;
    let training_speakers: BTreeSet<_> =
        training.iter().map(|pair| pair.speaker.as_str()).collect();
    for pair in &test {
        ensure!(
            !training_speakers.contains(pair.speaker.as_str()),
            "说话人 {} 同时出现在训练语料 {} 和官方测试集 {} 中（ID {}），无法隔离测试集",
            pair.speaker,
            root.join("noisy_trainset_28spk_wav").display(),
            pair.noisy.display(),
            pair.id
        );
    }
    let requested: BTreeSet<_> = validation_speakers.iter().cloned().collect();
    ensure!(
        !requested.is_empty(),
        "{}：未指定验证集说话人，请至少选择一名训练集说话人",
        root.display()
    );
    for speaker in &requested {
        ensure!(
            training_speakers.contains(speaker.as_str()),
            "{}：训练集 WAV 中不存在指定的验证集说话人 {}",
            root.display(),
            speaker
        );
    }
    let (validation, train): (Vec<_>, Vec<_>) = training
        .into_iter()
        .partition(|pair| requested.contains(&pair.speaker));
    ensure!(
        !train.is_empty(),
        "{}：留出验证集说话人后训练集为空，请减少验证集说话人",
        root.display()
    );
    ensure!(
        !validation.is_empty(),
        "{}：验证集为空，无法执行验证",
        root.display()
    );
    Ok(Corpus {
        train,
        validation,
        test,
        validation_speakers: requested.into_iter().collect(),
    })
}

fn wav_files(directory: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let mut files = BTreeMap::new();
    let entries = fs::read_dir(directory)
        .with_context(|| format!("无法读取 WAV 目录 {}", directory.display()))?;
    for entry in entries {
        let entry =
            entry.with_context(|| format!("无法读取目录 {} 中的条目", directory.display()))?;
        let path = entry.path();
        if !path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
        {
            continue;
        }
        ensure!(
            entry
                .file_type()
                .with_context(|| format!("无法读取 {} 的文件类型", path.display()))?
                .is_file(),
            "{}：WAV 条目不是普通文件",
            path.display()
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("{}：WAV 文件名不是有效的 UTF-8 文本", path.display()))?;
        files.insert(name, path);
    }
    Ok(files)
}

fn discover_pairs(noisy_directory: &Path, clean_directory: &Path) -> Result<Vec<Pair>> {
    let noisy_files = wav_files(noisy_directory)?;
    let mut clean_files = wav_files(clean_directory)?;
    let mut pairs = Vec::with_capacity(noisy_files.len());
    for (name, noisy) in noisy_files {
        let clean = clean_files.remove(&name).with_context(|| {
            format!(
                "{}：缺少同名的干净语音配对文件 {}",
                noisy.display(),
                clean_directory.join(&name).display()
            )
        })?;
        let id = noisy
            .file_stem()
            .and_then(|stem| stem.to_str())
            .with_context(|| format!("{}：无法从 WAV 文件名读取有效 ID", noisy.display()))?
            .to_owned();
        let (speaker, utterance) = id.split_once('_').with_context(|| {
            format!(
                "{}：ID {id} 缺少分隔说话人与语句编号的下划线",
                noisy.display()
            )
        })?;
        ensure!(
            !speaker.is_empty() && !utterance.is_empty(),
            "{}：ID {} 的说话人或语句编号为空",
            noisy.display(),
            id
        );
        let noisy_samples = inspect_wav(&noisy)?;
        let clean_samples = inspect_wav(&clean)?;
        ensure!(
            noisy_samples == clean_samples,
            "ID {}：配对文件长度不一致：{} 有 {} 个采样点；{} 有 {} 个采样点",
            id,
            noisy.display(),
            noisy_samples,
            clean.display(),
            clean_samples
        );
        let speaker = speaker.to_owned();
        pairs.push(Pair {
            id,
            speaker,
            noisy,
            clean,
            samples: noisy_samples,
        });
    }
    if let Some((name, clean)) = clean_files.into_iter().next() {
        bail!(
            "{}：缺少同名的含噪语音配对文件 {}",
            clean.display(),
            noisy_directory.join(name).display()
        );
    }
    ensure!(
        !pairs.is_empty(),
        "{} 与 {} 中没有可配对的 WAV 文件",
        noisy_directory.display(),
        clean_directory.display()
    );
    Ok(pairs)
}

type AudioReader = WavReader<BufReader<File>>;

fn open_wav(path: &Path) -> Result<AudioReader> {
    let reader = WavReader::open(path)
        .with_context(|| format!("无法打开或解析 WAV 文件头 {}", path.display()))?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1
            && spec.sample_rate == 48_000
            && spec.sample_format == SampleFormat::Int
            && spec.bits_per_sample == 16,
        "{}：只支持单声道、48000 Hz、PCM16 WAV；实际为 {} 声道、{} Hz、{:?}、{} 位",
        path.display(),
        spec.channels,
        spec.sample_rate,
        spec.sample_format,
        spec.bits_per_sample
    );
    ensure!(
        reader.duration() > 0,
        "{}：WAV 音频数据为空",
        path.display()
    );
    Ok(reader)
}

// Seeking to the declared data end does not prove those bytes exist: normal
// files allow seeking past EOF. Compare the logical end with the physical file
// length, including for crops that do not themselves touch a truncated tail.
fn check_extent(mut reader: AudioReader, path: &Path) -> Result<()> {
    reader
        .seek(reader.duration())
        .with_context(|| format!("无法定位到 WAV 文件 {} 声明的数据末尾", path.display()))?;
    let mut input = reader.into_inner();
    let declared_end = input
        .stream_position()
        .with_context(|| format!("无法读取 WAV 文件 {} 的当前位置", path.display()))?;
    let physical_end = input
        .get_ref()
        .metadata()
        .with_context(|| format!("无法读取 WAV 文件 {} 的实际长度", path.display()))?
        .len();
    ensure!(
        declared_end <= physical_end,
        "{}：WAV 数据已截断；文件头声明的数据末尾为第 {} 字节，实际文件仅有 {} 字节",
        path.display(),
        declared_end,
        physical_end
    );
    Ok(())
}

fn inspect_wav(path: &Path) -> Result<usize> {
    let mut reader = open_wav(path)?;
    let samples = reader.duration() as usize;
    // In addition to the public bit-depth spec, decoding one sample checks
    // that hound supports the actual PCM16 storage representation.
    reader
        .samples::<i16>()
        .next()
        .context("WAV 文件头声明音频非空，但无法读取首个采样点")
        .with_context(|| format!("无法读取 {} 的首个 PCM16 采样点", path.display()))?
        .with_context(|| format!("无法解码 {} 的首个 PCM16 采样点", path.display()))?;
    check_extent(reader, path)?;
    Ok(samples)
}

pub(super) fn read_segment(
    pair: &Pair,
    start: usize,
    noisy: &mut [f32],
    clean: &mut [f32],
) -> Result<usize> {
    ensure!(
        !noisy.is_empty() && noisy.len() == clean.len(),
        "ID {}（{} / {}）：裁剪缓冲区必须等长且非空，实际长度分别为 {} 和 {}",
        pair.id,
        pair.noisy.display(),
        pair.clean.display(),
        noisy.len(),
        clean.len()
    );
    ensure!(
        start <= pair.samples,
        "ID {}（{} / {}）：裁剪起点 {} 超出配对音频的 {} 个采样点",
        pair.id,
        pair.noisy.display(),
        pair.clean.display(),
        start,
        pair.samples
    );
    let valid = noisy.len().min(pair.samples - start);
    read_channel(&pair.noisy, pair.samples, start, valid, noisy)?;
    read_channel(&pair.clean, pair.samples, start, valid, clean)?;
    Ok(valid)
}

fn read_channel(
    path: &Path,
    expected_samples: usize,
    start: usize,
    valid: usize,
    output: &mut [f32],
) -> Result<()> {
    let mut reader = open_wav(path)?;
    ensure!(
        reader.duration() as usize == expected_samples,
        "{}：WAV 长度在语料发现后发生变化；预期 {} 个采样点，当前文件头声明 {} 个",
        path.display(),
        expected_samples,
        reader.duration()
    );
    let start = u32::try_from(start).with_context(|| {
        format!(
            "{}：裁剪起点 {start} 超出 WAV 支持的定位范围",
            path.display()
        )
    })?;
    reader
        .seek(start)
        .with_context(|| format!("{}：无法定位到第 {start} 个采样点", path.display()))?;
    let mut samples = reader.samples::<i16>();
    for (offset, target) in output[..valid].iter_mut().enumerate() {
        let sample = samples
            .next()
            .with_context(|| {
                format!(
                    "{}：WAV 数据在第 {} 个采样点处提前结束",
                    path.display(),
                    u64::from(start) + offset as u64
                )
            })?
            .with_context(|| {
                format!(
                    "{}：第 {} 个采样点的 PCM16 数据已截断或格式无效",
                    path.display(),
                    u64::from(start) + offset as u64
                )
            })?;
        *target = f32::from(sample) / 32768.0;
    }
    check_extent(reader, path)?;
    output[valid..].fill(0.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{SampleFormat, WavSpec, WavWriter};
    use std::fs;
    use tempfile::TempDir;

    const TRAIN_CLEAN: &str = "clean_trainset_28spk_wav";
    const TRAIN_NOISY: &str = "noisy_trainset_28spk_wav";
    const TEST_CLEAN: &str = "clean_testset_wav";
    const TEST_NOISY: &str = "noisy_testset_wav";

    struct Fixture(TempDir);

    impl Fixture {
        fn new() -> Self {
            let fixture = Self(tempfile::tempdir().unwrap());
            for directory in [TRAIN_CLEAN, TRAIN_NOISY, TEST_CLEAN, TEST_NOISY] {
                fs::create_dir(fixture.0.path().join(directory)).unwrap();
            }
            fixture.pair(false, "p225_002", &[100, 200, 300, 400]);
            fixture.pair(false, "p225_001", &[100, 200, 300, 400]);
            fixture.pair(false, "p226_001", &[100, 200, 300, 400]);
            fixture.pair(false, "p287_001", &[100, 200, 300, 400]);
            fixture.pair(true, "p232_001", &[100, 200, 300, 400]);
            fixture
        }

        fn path(&self, directory: &str, id: &str) -> std::path::PathBuf {
            self.0.path().join(directory).join(format!("{id}.wav"))
        }

        fn pair(&self, test: bool, id: &str, samples: &[i16]) {
            let directories = if test {
                [TEST_CLEAN, TEST_NOISY]
            } else {
                [TRAIN_CLEAN, TRAIN_NOISY]
            };
            for directory in directories {
                pcm16(&self.path(directory, id), samples, 48_000, 1);
            }
        }

        fn discover(&self) -> anyhow::Result<Corpus> {
            discover(self.0.path(), &["p226".into(), "p287".into()])
        }
    }

    fn pcm16(path: &std::path::Path, samples: &[i16], sample_rate: u32, channels: u16) {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        for &sample in samples {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn rejects_unmatched_filenames_in_either_official_split() {
        for directory in [TRAIN_CLEAN, TRAIN_NOISY, TEST_CLEAN, TEST_NOISY] {
            let fixture = Fixture::new();
            fixture.discover().unwrap();
            let missing_id = if directory == TEST_CLEAN || directory == TEST_NOISY {
                "p232_001"
            } else {
                "p225_001"
            };
            let path = fixture.path(directory, missing_id);
            fs::remove_file(&path).unwrap();
            assert!(fixture.discover().is_err());
        }
    }

    #[test]
    fn splits_whole_speakers_deterministically_and_isolates_official_test() {
        let fixture = Fixture::new();
        let corpus = fixture.discover().unwrap();
        assert_eq!(
            corpus
                .train
                .iter()
                .map(|pair| pair.id.as_str())
                .collect::<Vec<_>>(),
            ["p225_001", "p225_002"]
        );
        assert_eq!(
            corpus
                .validation
                .iter()
                .map(|pair| pair.id.as_str())
                .collect::<Vec<_>>(),
            ["p226_001", "p287_001"]
        );
        assert_eq!(corpus.validation_speakers, ["p226", "p287"]);
        assert_eq!(corpus.test.len(), 1);
        assert_eq!(corpus.test[0].speaker, "p232");
        assert_eq!(corpus.test[0].samples, 4);
        let reverse = discover(fixture.0.path(), &["p287".into(), "p226".into()]).unwrap();
        assert_eq!(reverse.validation_speakers, corpus.validation_speakers);
        assert_eq!(
            reverse
                .validation
                .iter()
                .map(|pair| &pair.id)
                .collect::<Vec<_>>(),
            corpus
                .validation
                .iter()
                .map(|pair| &pair.id)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn rejects_official_test_speaker_leakage_including_validation() {
        for speaker in ["p225", "p226"] {
            let fixture = Fixture::new();
            fixture.discover().unwrap();
            fixture.pair(true, &format!("{speaker}_999"), &[1, 2]);
            assert!(fixture.discover().is_err());
        }
    }

    #[test]
    fn rejects_absent_validation_speakers_and_empty_splits() {
        let fixture = Fixture::new();
        fixture.discover().unwrap();
        assert!(discover(fixture.0.path(), &["p999".into()]).is_err());
        assert!(discover(fixture.0.path(), &[]).is_err());
        assert!(
            discover(
                fixture.0.path(),
                &["p225".into(), "p226".into(), "p287".into()],
            )
            .is_err()
        );
        for directory in [TEST_CLEAN, TEST_NOISY] {
            fs::remove_file(fixture.path(directory, "p232_001")).unwrap();
        }
        assert!(fixture.discover().is_err());
    }

    #[test]
    fn reads_shared_crop_with_pcm_scaling_and_only_true_tail_padding() {
        let fixture = Fixture::new();
        pcm16(
            &fixture.path(TRAIN_NOISY, "p225_001"),
            &[111, i16::MIN, 16_384, i16::MAX],
            48_000,
            1,
        );
        pcm16(
            &fixture.path(TRAIN_CLEAN, "p225_001"),
            &[222, -16_384, 8_192, 0],
            48_000,
            1,
        );
        let corpus = fixture.discover().unwrap();
        let pair = &corpus.train[0];
        let mut noisy = [9.0; 5];
        let mut clean = [9.0; 5];
        assert_eq!(read_segment(pair, 1, &mut noisy, &mut clean).unwrap(), 3);
        assert_eq!(noisy, [-1.0, 0.5, 32767.0 / 32768.0, 0.0, 0.0]);
        assert_eq!(clean, [-0.5, 0.25, 0.0, 0.0, 0.0]);
        assert_eq!(
            read_segment(pair, 2, &mut noisy[..1], &mut clean[..1]).unwrap(),
            1
        );
        assert_eq!(noisy[0], 0.5);
        assert_eq!(clean[0], 0.25);
        assert_eq!(read_segment(pair, 4, &mut noisy, &mut clean).unwrap(), 0);
        assert_eq!(noisy, [0.0; 5]);
        assert_eq!(clean, [0.0; 5]);
        assert!(read_segment(pair, 5, &mut noisy, &mut clean).is_err());
        assert!(read_segment(pair, 0, &mut noisy[..0], &mut clean[..0]).is_err());
        assert!(read_segment(pair, 0, &mut noisy[..1], &mut clean[..2]).is_err());
    }

    #[test]
    fn rejects_bad_format_empty_audio_and_unequal_lengths() {
        for (rate, channels, samples) in [
            (16_000, 1, vec![1, 2, 3, 4]),
            (48_000, 2, vec![1, 2, 3, 4]),
            (48_000, 1, vec![]),
            (48_000, 1, vec![1, 2]),
        ] {
            let fixture = Fixture::new();
            fixture.discover().unwrap();
            let path = fixture.path(TRAIN_CLEAN, "p225_001");
            pcm16(&path, &samples, rate, channels);
            assert!(fixture.discover().is_err());
        }
        for (format, bits) in [(SampleFormat::Float, 32), (SampleFormat::Int, 24)] {
            let fixture = Fixture::new();
            fixture.discover().unwrap();
            let path = fixture.path(TRAIN_CLEAN, "p225_001");
            let mut writer = WavWriter::create(
                &path,
                WavSpec {
                    channels: 1,
                    sample_rate: 48_000,
                    bits_per_sample: bits,
                    sample_format: format,
                },
            )
            .unwrap();
            if format == SampleFormat::Float {
                writer.write_sample(0.5_f32).unwrap();
            } else {
                writer.write_sample(100_i32).unwrap();
            }
            writer.finalize().unwrap();
            assert!(fixture.discover().is_err());
        }
    }

    #[test]
    fn rejects_truncated_data_at_discovery_and_after_discovery_even_outside_crop() {
        let fixture = Fixture::new();
        let corpus = fixture.discover().unwrap();
        let pair = &corpus.train[0];
        let mut noisy = [0.0; 1];
        let mut clean = [0.0; 1];
        assert_eq!(read_segment(pair, 0, &mut noisy, &mut clean).unwrap(), 1);
        assert_eq!(noisy, [100.0 / 32768.0]);
        assert_eq!(clean, [100.0 / 32768.0]);
        let file = fs::OpenOptions::new()
            .write(true)
            .open(&pair.clean)
            .unwrap();
        file.set_len(file.metadata().unwrap().len() - 2).unwrap();
        drop(file);
        assert!(fixture.discover().is_err());
        assert!(read_segment(pair, 0, &mut noisy, &mut clean).is_err());
        assert!(read_segment(pair, pair.samples, &mut noisy, &mut clean).is_err());
    }

    #[test]
    fn rejects_changed_pair_headers_instead_of_padding_or_misaligning() {
        let fixture = Fixture::new();
        let corpus = fixture.discover().unwrap();
        let pair = &corpus.train[0];
        let mut noisy = [0.0; 4];
        let mut clean = [0.0; 4];
        assert_eq!(read_segment(pair, 0, &mut noisy, &mut clean).unwrap(), 4);
        assert_eq!(
            clean,
            [
                100.0 / 32768.0,
                200.0 / 32768.0,
                300.0 / 32768.0,
                400.0 / 32768.0
            ]
        );
        assert_eq!(noisy, clean);
        pcm16(&pair.clean, &[1, 2], 48_000, 1);
        assert!(read_segment(pair, 0, &mut noisy, &mut clean).is_err());
    }
}
