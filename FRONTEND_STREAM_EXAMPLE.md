# Frontend usage for streamed video

This backend returns `video/mp4` from `POST /text-to-sign`.
It does not return JSON. The generated pose file URL is returned in the response header:

`X-Pose-URL`

Use `fetch`, convert the response to a Blob, then assign it to a video element.

```js
async function generateSignVideo(sentence) {
  const res = await fetch("http://text-to-sign-api-isharati.us-east-1.elasticbeanstalk.com/text-to-sign", {
    method: "POST",
    headers: {
      "x-api-key": "texttosign123456",
      "Content-Type": "application/json; charset=utf-8"
    },
    body: JSON.stringify({ sentence })
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }

  const poseUrl = res.headers.get("X-Pose-URL");
  const videoBlob = await res.blob();
  const videoUrl = URL.createObjectURL(videoBlob);

  return { videoUrl, poseUrl };
}
```

```jsx
const result = await generateSignVideo("انا اسمي توماس");
setVideoUrl(result.videoUrl);
setPoseUrl(result.poseUrl);
```

```jsx
{videoUrl && (
  <video
    src={videoUrl}
    controls
    autoPlay
    style={{ width: "100%", maxWidth: 700 }}
  />
)}
```

Do not use `<video src="/text-to-sign">` directly because the route requires POST body + API key.
