extern crate pretty_env_logger;
#[macro_use]
extern crate log;
mod frame;
mod pipeline;
mod stream;

use std::env;

use frame::Frame;
use pipeline::Pipeline;
use stream::VideoStream;

async fn process_frame(frame: Frame) -> Frame {
    let processed_frame = frame
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

    // debug!(
    //     "processing frame {} on thread {} of {}",
    //     frame.num,
    //     current_thread_index().unwrap(),
    //     current_num_threads(),
    // );

    //
    processed_frame
}

async fn route_frames(pipe: Pipeline) -> ! {
    //let mut output_frames: HashMap<i32, Frame> = HashMap::new();

    loop {
        tokio::select! {
            frame = pipe.decode_recv() => {
                let f = frame.unwrap();
                debug!("frame {}\tdecoded", f.num);
                pipe.process_send(f).await;
            },
            frame = pipe.process_recv() => {
                let p = pipe.clone();
                tokio::spawn(async move {
                    let f = frame.unwrap();
                    let frame_num = f.num;
                    let processed_frame = process_frame(f).await;
                    debug!("frame {}\tprocessed", frame_num);
                    p.output_send(processed_frame).await;
                });
            },
            frame = pipe.output_recv() => {
                let f = frame.unwrap();
                debug!("frame {}\toutput", f.num);
                //output_frames.insert(f.num, f);
                // let mut sorted: Vec<_> = output_frames.iter().collect();
                // sorted.sort_by_key(|a| a.0);
                //
                // let (frame_num, current_frame) = sorted.pop().unwrap();
                // imshow("frames", &current_frame.processed_mat).unwrap();
                // poll_key().unwrap();
            },

        }
    }
}

#[tokio::main]
async fn main() {
    pretty_env_logger::init_timed();
    let url = &env::args().nth(1).expect("cannot open");
    let pipe = pipeline::new();
    let p = pipe.clone();

    tokio::spawn(async move {
        route_frames(p).await;
    });

    let mut stream = VideoStream::new(url.to_string(), &pipe);
    stream.decode().await
}
