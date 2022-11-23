use clap::Parser;

extern crate pretty_env_logger;
#[macro_use]
extern crate log;

pub mod games;
mod img;
pub mod pipeline;
mod roi;
mod state;
pub mod stream;

/// Stream Processor
#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
pub struct Config {
    /// The URL of the stream.
    #[clap(short, long)]
    pub url: String,

    /// Show processed frames in a window
    #[clap(short, long)]
    pub show_frames: bool,

    #[clap(long, default_value_t = 0)]
    pub num_opencv_threads: i32,

    #[clap(long, default_value_t = 1)]
    pub num_libav_threads: i32,

    #[clap(long, default_value_t = 2)]
    pub process_frame_rate: i64,
}
