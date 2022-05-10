extern crate pretty_env_logger;
#[macro_use]
extern crate log;
mod frame;
mod pipeline;
mod stream;

use std::env;

use frame::Frame;
use futures::StreamExt;
use pipeline::Pipeline;
use stream::VideoStream;
use tokio::select;

async fn process_frame(frame: Frame) -> Frame {
    let processed_frame = frame
        .bilateral_filter()
        .await
        .detail_enhance()
        .await
        .dilate()
        .await;

    processed_frame
}

async fn route_frames(pipe: Pipeline) -> ! {
    let mut decode_stream = pipe.decode_stream();
    let mut process_stream = pipe.process_stream();
    let mut output_stream = pipe.output_stream();

    loop {
        select! {
            frame = decode_stream.next() => {
                let f = frame.unwrap();
                debug!("frame {}\tdecoded", f.num);
                pipe.process_send(f).await;
            },
            frame = process_stream.next() => {
                let p = pipe.clone();
                tokio::spawn(async move {
                    let f = frame.unwrap();
                    let frame_num = f.num;
                    let processed_frame = process_frame(f).await;
                    debug!("frame {}\tprocessed", frame_num);
                    p.output_send(processed_frame).await;
                });
            },
            frame = output_stream.next() => {
                let f = frame.unwrap();
                debug!("frame {}\toutput", f.num);
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
