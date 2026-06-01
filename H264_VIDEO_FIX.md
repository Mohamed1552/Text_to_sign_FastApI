# H.264 video fix

The old generated MP4 used `mp4v` codec from OpenCV/PoseVisualizer.
Many browsers, Windows players, and Postman preview do not support this codec.

This version:
1. Generates the raw pose video.
2. Converts it to browser-compatible H.264 MP4 using imageio-ffmpeg.
3. Returns the H.264 MP4 as the API response.
4. Deletes the temporary raw file.

Expected codec after fix:
- Container: MP4
- Video codec: H.264 / avc1
- Pixel format: yuv420p
