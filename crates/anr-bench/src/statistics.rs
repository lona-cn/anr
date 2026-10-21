use anr_dfn3::{FRAME_SIZE, SAMPLE_RATE};
use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Serialize)]
pub struct Statistics {
    pub elapsed_ms: f64,
    pub audio_seconds: f64,
    pub min_ms: f64,
    pub mean_ms: f64,
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub frames_per_second: f64,
    pub rtf: f64,
    pub over_budget_frames: usize,
}

impl Statistics {
    // The CLI requires at least one measured frame. Sorting happens after timing.
    pub fn calculate(latencies: &mut [Duration], elapsed: Duration) -> Self {
        let count = latencies.len();
        let frame_seconds = FRAME_SIZE as f64 / f64::from(SAMPLE_RATE);
        let budget = Duration::from_secs_f64(frame_seconds);
        let over_budget_frames = latencies.iter().filter(|&&time| time > budget).count();
        let mean_ms =
            latencies.iter().map(Duration::as_secs_f64).sum::<f64>() * 1_000.0 / count as f64;
        latencies.sort_unstable();
        // Nearest rank: ceil(N * percentile / 100), with one-based ranks.
        let percentile = |percent: u64| {
            latencies[((count as u64 * percent).div_ceil(100) - 1) as usize].as_secs_f64() * 1_000.0
        };
        let audio_seconds = count as f64 * frame_seconds;
        let elapsed_seconds = elapsed.as_secs_f64();
        Self {
            elapsed_ms: elapsed_seconds * 1_000.0,
            audio_seconds,
            min_ms: latencies[0].as_secs_f64() * 1_000.0,
            mean_ms,
            max_ms: latencies[count - 1].as_secs_f64() * 1_000.0,
            p50_ms: percentile(50),
            p95_ms: percentile(95),
            p99_ms: percentile(99),
            frames_per_second: count as f64 / elapsed_seconds,
            rtf: elapsed_seconds / audio_seconds,
            over_budget_frames,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Statistics;
    use std::time::Duration;

    #[test]
    fn reports_nearest_rank_tails_and_counts_only_frames_over_the_deadline() {
        let mut latencies = [11, 1, 10, 2, 9, 3, 8, 4, 7, 5].map(Duration::from_millis);
        let stats = Statistics::calculate(&mut latencies, Duration::from_millis(80));

        assert_eq!(stats.min_ms, 1.0);
        assert_eq!(stats.max_ms, 11.0);
        assert!((stats.mean_ms - 6.0).abs() < 1e-9);
        assert_eq!(stats.p50_ms, 5.0);
        assert_eq!(stats.p95_ms, 11.0);
        assert_eq!(stats.p99_ms, 11.0);
        assert_eq!(stats.over_budget_frames, 1);
    }

    #[test]
    fn derives_throughput_and_rtf_from_wall_time_not_the_sum_of_call_latencies() {
        let mut latencies = [Duration::from_millis(2), Duration::from_millis(4)];
        let stats = Statistics::calculate(&mut latencies, Duration::from_millis(10));

        assert_eq!(stats.mean_ms, 3.0);
        assert_eq!(stats.elapsed_ms, 10.0);
        assert_eq!(stats.audio_seconds, 0.02);
        assert_eq!(stats.frames_per_second, 200.0);
        assert_eq!(stats.rtf, 0.5);
    }

    #[test]
    fn supports_a_single_measured_frame_without_percentile_underflow() {
        let mut latencies = [Duration::from_millis(10)];
        let stats = Statistics::calculate(&mut latencies, Duration::from_millis(10));

        assert_eq!(stats.min_ms, 10.0);
        assert_eq!(stats.mean_ms, 10.0);
        assert_eq!(stats.max_ms, 10.0);
        assert_eq!(stats.p50_ms, 10.0);
        assert_eq!(stats.p95_ms, 10.0);
        assert_eq!(stats.p99_ms, 10.0);
        assert_eq!(stats.over_budget_frames, 0);
        assert_eq!(stats.rtf, 1.0);
    }
}
