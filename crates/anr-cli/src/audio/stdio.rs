use std::{
    io::{self, Read, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError},
    },
    thread,
    time::Duration,
};

const BUFFER_BYTES: usize = 64 * 1024;
const WAIT: Duration = Duration::from_millis(25);
pub(super) fn stdout_file() -> io::Result<std::fs::File> {
    let stdout = std::io::stdout();
    // Duplicate the OS handle, not Stdout's process-global LineWriter. Otherwise
    // a short tail can remain there and block the main thread during exit cleanup.
    #[cfg(unix)]
    let owned = {
        use std::os::fd::AsFd;
        stdout.as_fd().try_clone_to_owned()?
    };
    #[cfg(windows)]
    let owned = {
        use std::os::windows::io::AsHandle;
        stdout.as_handle().try_clone_to_owned()?
    };
    Ok(std::fs::File::from(owned))
}

fn check(cancelled: &AtomicBool) -> io::Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(io::Error::other(crate::cancel::Cancelled));
    }
    Ok(())
}

fn send<T>(tx: &SyncSender<T>, mut value: T, cancelled: &AtomicBool) -> io::Result<()> {
    loop {
        check(cancelled)?;
        match tx.try_send(value) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(returned)) => {
                value = returned;
                thread::sleep(WAIT);
            }
            Err(TrySendError::Disconnected(_)) => {
                check(cancelled)?;
                return Err(io::Error::other("stdio worker disconnected"));
            }
        }
    }
}

fn receive<T>(rx: &Receiver<T>, cancelled: &AtomicBool) -> io::Result<T> {
    loop {
        check(cancelled)?;
        match rx.recv_timeout(WAIT) {
            Ok(value) => return Ok(value),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                check(cancelled)?;
                return Err(io::Error::other("stdio worker disconnected"));
            }
        }
    }
}

// A single reusable buffer crosses each bounded request/reply pair. A blocked OS
// operation may outlive cancellation, but owns only the standard stream, never
// an output transaction. Dropping the caller disconnects the next request.
pub(super) struct CancellableReader {
    requests: SyncSender<Vec<u8>>,
    replies: Receiver<(Vec<u8>, io::Result<usize>)>,
    bytes: Vec<u8>,
    position: usize,
    valid: usize,
    eof: bool,
    cancelled: Arc<AtomicBool>,
}

impl CancellableReader {
    pub(super) fn spawn<R: Read + Send + 'static>(
        mut reader: R,
        cancelled: Arc<AtomicBool>,
    ) -> io::Result<Self> {
        let (requests, request_rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let (replies_tx, replies) = mpsc::sync_channel(1);
        let worker_cancelled = cancelled.clone();
        thread::Builder::new()
            .name("anr-stdin".into())
            .spawn(move || {
                while let Ok(mut bytes) = receive(&request_rx, &worker_cancelled) {
                    let result = loop {
                        if let Err(error) = check(&worker_cancelled) {
                            break Err(error);
                        }
                        match reader.read(&mut bytes) {
                            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                            result => break result,
                        }
                    };
                    if send(&replies_tx, (bytes, result), &worker_cancelled).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self {
            requests,
            replies,
            bytes: vec![0; BUFFER_BYTES],
            position: 0,
            valid: 0,
            eof: false,
            cancelled,
        })
    }
}

impl Read for CancellableReader {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        check(&self.cancelled)?;
        if output.is_empty() || self.eof {
            return Ok(0);
        }
        if self.position == self.valid {
            send(
                &self.requests,
                std::mem::take(&mut self.bytes),
                &self.cancelled,
            )?;
            let (bytes, result) = receive(&self.replies, &self.cancelled)?;
            self.bytes = bytes;
            self.valid = result?;
            self.position = 0;
            if self.valid == 0 {
                self.eof = true;
                return Ok(0);
            }
        }
        let count = output.len().min(self.valid - self.position);
        output[..count].copy_from_slice(&self.bytes[self.position..self.position + count]);
        self.position += count;
        Ok(count)
    }
}

enum WriteRequest {
    Bytes(Vec<u8>),
    Flush,
}

pub(super) struct CancellableWriter {
    requests: SyncSender<WriteRequest>,
    replies: Receiver<(Vec<u8>, io::Result<()>)>,
    cancelled: Arc<AtomicBool>,
}

impl CancellableWriter {
    pub(super) fn spawn<W: Write + Send + 'static>(
        mut writer: W,
        cancelled: Arc<AtomicBool>,
    ) -> io::Result<Self> {
        let (requests, request_rx) = mpsc::sync_channel(1);
        let (replies_tx, replies) = mpsc::sync_channel(1);
        let worker_cancelled = cancelled.clone();
        thread::Builder::new()
            .name("anr-stdout".into())
            .spawn(move || {
                while let Ok(request) = receive(&request_rx, &worker_cancelled) {
                    let (bytes, result) = match request {
                        WriteRequest::Bytes(bytes) => {
                            let result = writer.write_all(&bytes);
                            (bytes, result)
                        }
                        WriteRequest::Flush => (Vec::new(), writer.flush()),
                    };
                    if send(&replies_tx, (bytes, result), &worker_cancelled).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self {
            requests,
            replies,
            cancelled,
        })
    }
    pub(super) fn write_buffer(&mut self, buffer: &mut Vec<u8>) -> io::Result<()> {
        check(&self.cancelled)?;
        if buffer.is_empty() {
            return Ok(());
        }
        send(
            &self.requests,
            WriteRequest::Bytes(std::mem::take(buffer)),
            &self.cancelled,
        )?;
        let (bytes, result) = receive(&self.replies, &self.cancelled)?;
        *buffer = bytes;
        result
    }

    pub(super) fn flush(&mut self) -> io::Result<()> {
        check(&self.cancelled)?;
        send(&self.requests, WriteRequest::Flush, &self.cancelled)?;
        let (_, result) = receive(&self.replies, &self.cancelled)?;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Condvar, Mutex};

    struct Blocked(Arc<(Mutex<bool>, Condvar)>);
    impl Read for Blocked {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            let (lock, wake) = &*self.0;
            let _guard = wake
                .wait_while(lock.lock().unwrap(), |done| !*done)
                .unwrap();
            Ok(0)
        }
    }
    impl Write for Blocked {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.read(&mut [])?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    fn release(gate: &Arc<(Mutex<bool>, Condvar)>) {
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
    }
    fn cancel_soon(flag: Arc<AtomicBool>) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            flag.store(true, Ordering::Relaxed);
        })
    }
    fn unblock_watchdog(
        gate: Arc<(Mutex<bool>, Condvar)>,
    ) -> (mpsc::Sender<()>, thread::JoinHandle<bool>) {
        let (finished, completion) = mpsc::channel();
        let watchdog = thread::spawn(move || {
            let timed_out = completion.recv_timeout(Duration::from_secs(2)).is_err();
            release(&gate);
            timed_out
        });
        (finished, watchdog)
    }
    #[test]
    fn blocked_reader_returns_cancel_marker_without_waiting_for_stdin() {
        let flag = Arc::new(AtomicBool::new(false));
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let mut reader = CancellableReader::spawn(Blocked(gate.clone()), flag.clone()).unwrap();
        let trigger = cancel_soon(flag);
        let (finished, watchdog) = unblock_watchdog(gate);
        let result = reader.read(&mut [0; 4]);
        let _ = finished.send(());
        let timed_out = watchdog.join().unwrap();
        trigger.join().unwrap();
        assert!(
            !timed_out,
            "stdin cancellation did not return before the watchdog released IO"
        );
        assert!(
            result
                .unwrap_err()
                .get_ref()
                .unwrap()
                .is::<crate::cancel::Cancelled>()
        );
    }
    #[test]
    fn blocked_writer_returns_cancel_marker_without_waiting_for_consumer() {
        let flag = Arc::new(AtomicBool::new(false));
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let mut writer = CancellableWriter::spawn(Blocked(gate.clone()), flag.clone()).unwrap();
        let trigger = cancel_soon(flag);
        let (finished, watchdog) = unblock_watchdog(gate);
        let result = writer.write_buffer(&mut vec![1, 2, 3, 4]);
        let _ = finished.send(());
        let timed_out = watchdog.join().unwrap();
        trigger.join().unwrap();
        assert!(
            !timed_out,
            "stdout cancellation did not return before the watchdog released IO"
        );
        assert!(
            result
                .unwrap_err()
                .get_ref()
                .unwrap()
                .is::<crate::cancel::Cancelled>()
        );
    }
    #[test]
    fn writer_preserves_broken_pipe_kind() {
        struct Closed;
        impl Write for Closed {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut writer =
            CancellableWriter::spawn(Closed, Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(
            writer
                .write_buffer(&mut b"audio".to_vec())
                .unwrap_err()
                .kind(),
            io::ErrorKind::BrokenPipe
        );
    }
    #[test]
    fn stdout_small_tail_is_delivered_before_flush_or_process_exit() {
        const CHILD: &str = "ANR_STDIO_SMALL_TAIL_CHILD";
        const TOKEN: [u8; 4] = [0, 0xff, 0xfe, 0xfd];
        if std::env::var_os(CHILD).is_some() {
            let mut writer =
                CancellableWriter::spawn(stdout_file().unwrap(), Arc::new(AtomicBool::new(false)))
                    .unwrap();
            writer.write_buffer(&mut TOKEN.to_vec()).unwrap();
            // Parent holds stdin open until it observes the short, newline-free
            // payload. No flush or normal-exit stdio cleanup can deliver it.
            std::io::stdin().read_exact(&mut [0]).unwrap_err();
            return;
        }
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "audio::stdio::tests::stdout_small_tail_is_delivered_before_flush_or_process_exit",
                "--nocapture",
                "--quiet",
            ])
            .env(CHILD, "1")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let mut stdout = child.stdout.take().unwrap();
        let (observed, observation) = mpsc::channel();
        let reader = thread::spawn(move || {
            let mut tail = [0; TOKEN.len()];
            let mut delivered = false;
            let mut bytes = 0;
            loop {
                let mut byte = [0];
                match stdout.read(&mut byte) {
                    Ok(0) => break,
                    Ok(_) => {
                        tail.rotate_left(1);
                        tail[TOKEN.len() - 1] = byte[0];
                        bytes += 1;
                        if !delivered && bytes >= TOKEN.len() && tail == TOKEN {
                            let _ = observed.send(());
                            delivered = true;
                        }
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
        });
        let delivered = observation.recv_timeout(Duration::from_secs(3)).is_ok();
        drop(child.stdin.take());
        if !delivered {
            let _ = child.kill();
        }
        let status = child.wait().unwrap();
        reader.join().unwrap();
        assert!(
            delivered,
            "small stdout tail remained buffered while the process was alive"
        );
        assert!(status.success(), "stdout child failed: {status}");
    }
}
