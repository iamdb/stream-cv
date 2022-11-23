#[tokio::main]
async fn main() {
    stream_cv::start(stream_cv::games::SupportedGames::Apex).await;
}
