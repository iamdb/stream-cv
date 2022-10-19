extern crate ffmpeg_next as ffmpeg;

use crate::img::frame::Frame;
use crate::{roi, Config};
use chrono::Utc;
use ffmpeg::format::{input, Pixel};
use ffmpeg::frame::Video;
use ffmpeg::software::scaling::{context::Context as FFContext, flag::Flags};
use ffmpeg::sys::{av_log_set_level, AV_LOG_QUIET};
use flume::Sender;
use opencv::core::{Mat, ToOutputArray, UMat, CV_8UC3};
use opencv::prelude::MatTraitConst;
use std::ffi::c_void;

pub struct VideoStream {
    _width: i32,
    _height: i32,
    _framerate: i32,
    pub decoding: bool,
    pub frame_index: i64,
    pub frame_rate: i32,
    scaler: Option<FFContext>,
    sender: Sender<Frame>,
    config: Config,
}

impl VideoStream {
    pub fn new(config: Config, sender: Sender<Frame>) -> Self {
        VideoStream {
            _width: 0,
            _height: 0,
            _framerate: 0,
            decoding: false,
            scaler: None,
            frame_index: 0,
            sender,
            config,
            frame_rate: 0,
        }
    }
    pub async fn decode(&mut self) {
        ffmpeg::init().unwrap();

        unsafe { av_log_set_level(AV_LOG_QUIET) }

        info!("****setting up decoder*****");
        if let Ok(mut ictx) = input(&self.config.url) {
            info!("*****building input*****");
            let streams = ictx.streams();

            if let Some(input) = streams.best(ffmpeg::media::Type::Video) {
                let video_stream_index = input.index();
                if let Some(comment) = input.metadata().get("comment") {
                    info!("Stream spec: {}", comment);
                }

                let mut context_decoder =
                    ffmpeg::codec::context::Context::from_parameters(input.parameters()).unwrap();

                context_decoder.set_threading(ffmpeg::threading::Config {
                    count: self.config.num_libav_threads as usize,
                    kind: ffmpeg::threading::Type::Frame,
                    safe: true,
                });

                debug!("libav threading\t{:?}", context_decoder.threading());

                let mut decoder = context_decoder.decoder().video().unwrap();

                self.scaler = Some(
                    FFContext::get(
                        decoder.format(),
                        decoder.width(),
                        decoder.height(),
                        Pixel::BGR24,
                        decoder.width(),
                        decoder.height(),
                        Flags::LANCZOS,
                    )
                    .unwrap(),
                );

                info!("decoding stream");
                self.decoding = true;

                for (stream, packet) in ictx.packets() {
                    if stream.index() == video_stream_index {
                        match decoder.send_packet(&packet) {
                            Ok(_) => self.receive_and_process_decoded_frames(&mut decoder).await,
                            Err(error) => error!("{}", error.to_string()),
                        }
                    }
                }

                decoder.send_eof().unwrap();
                self.receive_and_process_decoded_frames(&mut decoder).await;

                self.decoding = false;
            } else {
                error!("no streams available");
            }
        }
    }

    async fn receive_and_process_decoded_frames(&mut self, decoder: &mut ffmpeg::decoder::Video) {
        let mut decoded = Video::empty();

        while decoder.receive_frame(&mut decoded).is_ok() {
            if self.frame_index.rem_euclid(self.config.process_frame_rate) != 0 {
                self.frame_index += 1;
                continue;
            }
            let mut rgb_frame = Video::empty();

            self.scaler
                .as_mut()
                .unwrap()
                .run(&decoded, &mut rgb_frame)
                .unwrap();

            let bgr_mat: Mat;
            let mut bgr_umat = UMat::new(opencv::core::UMatUsageFlags::USAGE_DEFAULT);

            unsafe {
                let raw_rgb_frame = rgb_frame.as_ptr().as_ref().unwrap();

                bgr_mat = Mat::new_rows_cols_with_data(
                    raw_rgb_frame.height,
                    raw_rgb_frame.width,
                    CV_8UC3,
                    raw_rgb_frame.data[0] as *mut c_void,
                    raw_rgb_frame.linesize[0].try_into().unwrap(),
                )
                .unwrap();
            }

            bgr_mat
                .copy_to(&mut bgr_umat.output_array().unwrap())
                .unwrap();

            let new_frame = crate::img::frame::Frame {
                mat: bgr_mat.clone(),
                num: self.frame_index,
                processed_mat: bgr_umat,
                text: "".to_string(),
                start_date: Utc::now(),
                end_date: None,
                results: roi::new_region_list(),
            };

            self.sender.send_async(new_frame).await.unwrap();

            self.frame_index += 1;
        }
    }
}
