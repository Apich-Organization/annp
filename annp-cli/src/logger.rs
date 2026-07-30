use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

pub struct AnnpLogger {
    file_writer: Option<Mutex<File>>,
    log_file_path: Option<PathBuf>,
}

impl AnnpLogger {
    pub fn new(log_dir: &Path, prefix: &str, custom_file: Option<&Path>) -> Self {
        if let Err(e) = fs::create_dir_all(log_dir) {
            eprintln!(
                "Warning: Could not create log directory {:?}: {}",
                log_dir, e
            );
        }

        let file_path = if let Some(p) = custom_file {
            p.to_path_buf()
        } else {
            let timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            log_dir.join(format!("{}_{}.log", prefix, timestamp))
        };

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&file_path)
            .map_err(|e| {
                eprintln!("Warning: Failed to open log file {:?}: {}", file_path, e);
                e
            })
            .ok();

        if file.is_some() {
            println!(
                "Logging detailed execution metrics to file: {:?}",
                file_path
            );
        }

        Self {
            file_writer: file.map(Mutex::new),
            log_file_path: Some(file_path),
        }
    }

    pub fn get_log_path(&self) -> Option<&PathBuf> {
        self.log_file_path.as_ref()
    }

    pub fn log(&self, tag: &str, msg: &str) {
        let timestamp = chrono_timestamp();
        let formatted = format!("[{}] {} | {}\n", timestamp, tag, msg);

        // Always print to console
        print!("{}", formatted);

        // Write to log file if available
        if let Some(ref writer_mutex) = self.file_writer {
            if let Ok(mut file) = writer_mutex.lock() {
                let _ = file.write_all(formatted.as_bytes());
                let _ = file.flush();
            }
        }
    }

    pub fn log_step(
        &self,
        epoch: usize,
        total_epochs: usize,
        batch_idx: usize,
        step_loss: f32,
        rolling_loss: f32,
    ) {
        let msg = format!(
            "[Epoch {:2}/{:2} | Batch {:4}] Step Loss: {:.6} | Rolling Loss: {:.6}",
            epoch, total_epochs, batch_idx, step_loss, rolling_loss
        );
        self.log("TRAIN_STEP", &msg);
    }

    pub fn log_hardening(
        &self,
        epoch: usize,
        links_before: usize,
        links_pruned: usize,
        nodes_grown: usize,
        spawn_details: &[(usize, usize, usize)], // (parent_a, parent_b, new_id)
    ) {
        let remaining_links = links_before.saturating_sub(links_pruned);
        let mut msg = format!(
            "Hardening Pass (Epoch {}): Pruned {} links (Remaining: {}). Spawned {} local subnodes.",
            epoch, links_pruned, remaining_links, nodes_grown
        );

        for (p_a, p_b, new_id) in spawn_details {
            msg.push_str(&format!(
                "\n  -> Node {}: subnodes {} -> {}",
                p_a, p_b, new_id
            ));
        }

        self.log("HARDENING", &msg);
    }
}

fn chrono_timestamp() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now();
    if let Ok(dur) = now.duration_since(SystemTime::UNIX_EPOCH) {
        let secs = dur.as_secs();
        let millis = dur.subsec_millis();
        let hours = (secs / 3600) % 24;
        let mins = (secs / 60) % 60;
        let s = secs % 60;
        format!("{:02}:{:02}:{:02}.{:03}", hours, mins, s, millis)
    } else {
        "00:00:00.000".to_string()
    }
}
