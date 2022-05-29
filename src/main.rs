extern crate pretty_env_logger;
#[macro_use]
extern crate log;

mod img;
mod pipeline;
mod roi;
mod stream;

use clap::Parser;
use img::frame::Frame;
use opencv::core::{get_num_threads, set_num_threads};
use stream::VideoStream;

/// Stream Processor
#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
pub struct Config {
    /// The URL of the stream.
    #[clap(short, long)]
    url: String,

    /// Show processed frames in a window
    #[clap(short, long)]
    show_frames: bool,

    #[clap(long, default_value_t = 1)]
    num_opencv_threads: i32,

    #[clap(long, default_value_t = 1)]
    num_libav_threads: i32,
}

#[tokio::main]
async fn main() {
    pretty_env_logger::init_timed();

    let config = Config::parse();

    if !opencv::core::use_optimized().expect("error checking for optimized code") {
        debug!("changing opencv to use optimized code");
        opencv::core::set_use_optimized(true).expect("error enabling optimized code");
    } else {
        debug!("opencv is using optimized code")
    }

    set_num_threads(config.num_opencv_threads).unwrap();

    let opencv_threads = get_num_threads().expect("error retrieving number of threads");
    debug!("opencv is using {} threads", opencv_threads);

    let regions = make_regions();

    let pipe = pipeline::new(regions, config.clone());
    pipe.start_router().await;

    let mut stream = VideoStream::new(config, &pipe);
    stream.decode().await
}

fn make_regions() -> roi::RegionOfInterestList {
    let mut list = roi::new_region_list();

    //list.add_new_region("materials_size".to_string(), 1520, 53, 100, 24);
    list.add_new_region("loaded_mag_size".to_string(), 1720, 960, 62, 40);
    list.add_new_region("total_ammo".to_string(), 1720, 998, 62, 30);

    list
}
