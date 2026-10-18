# Free media generation CI (ZEE demo)

The **Free image and video demo pipeline** runs daily at 08:00 Sri Lanka time and can also be started manually from GitHub Actions. It attempts a Pollinations image request, discovers candidate video files through Wikimedia Commons, and uses FFmpeg to render a short MP4 from the generated still image. Assets, candidate attribution metadata and a JSON run report are uploaded as a GitHub Actions artifact.

Run manually: **Actions → Free image and video demo pipeline → Run workflow**.

Limitations:
- The video is a six-second MP4 made from one generated still image, not a multi-shot video-generation model.
- Commons search is metadata-only. Third-party clips are not downloaded automatically. Review each file's page, licence, creator/attribution, commercial-use rights and ShareAlike requirements before use in ZEE.
- Pollinations is a third-party service; availability and usage terms can change. Check \`run-report.json\` to distinguish successful CI execution from successful image generation.
- This workflow does not train an image/video foundation model, publish a model, or deploy to the hosted ZEE site.
