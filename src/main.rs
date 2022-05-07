mod frame;
mod pipeline;
mod stream;

use std::env;

use frame::Frame;
use opencv::core::get_number_of_cpus;
use pipeline::Pipeline;
use stream::VideoStream;

async fn process_frame(frame: Frame) {
    frame
        .bilateral_filter()
        .await
        .detail_enhance()
        .await
        .dilate()
        .await;

    // let alpha = 1.0;
    // let beta = 0.25;
    //
    // let overlayed = frame.bilateral_filter().dilate().add_weighted(
    //     processed_frame.processed_mat.clone(),
    //     alpha,
    //     beta,
    // );

    // println!(
    //     "processing frame {} on thread {} of {}",
    //     frame.num,
    //     current_thread_index().unwrap(),
    //     current_num_threads(),
    // );

    //imshow("frames", &processed_frame.processed_mat).unwrap();
    //imshow("og", &frame.mat).unwrap();
    //poll_key().unwrap();
}

async fn process_thread(thread_num: i32, pipe: Pipeline) {
    tokio::spawn(async move {
        let frame_stream = pipe.stream();

        while !frame_stream.is_disconnected() {
            let frame = pipe.recv().await.unwrap();
            process_frame(frame.clone()).await;
            println!("processed frame {} on thread {}", frame.num, thread_num);
        }
    });
}

#[tokio::main]
async fn main() {
    let url = &env::args().nth(1).expect("cannot open");
    let num_of_cpus = get_number_of_cpus().unwrap();
    let pipe = pipeline::new();

    for n in 0..num_of_cpus {
        process_thread(n, pipe.clone()).await;
    }

    let mut stream = VideoStream::new(url.to_string(), &pipe);
    stream.decode().await
}
