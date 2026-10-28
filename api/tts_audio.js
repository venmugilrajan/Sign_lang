export const config = {
  runtime: 'edge',
};

export default async function handler(request) {
  const { searchParams } = new URL(request.url);
  const q = searchParams.get('q') || '';
  const tl = searchParams.get('tl') || 'en';

  if (!q.trim()) {
    return new Response('Missing text query', { status: 400 });
  }

  try {
    const googleTtsUrl = `https://translate.google.com/translate_tts?ie=UTF-8&tl=${encodeURIComponent(tl)}&client=tw-ob&q=${encodeURIComponent(q.trim())}`;
    const ttsRes = await fetch(googleTtsUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://translate.google.com/',
      },
    });

    if (!ttsRes.ok) {
      return new Response(`TTS upstream failed with status: ${ttsRes.status}`, { status: ttsRes.status });
    }

    const audioBuffer = await ttsRes.arrayBuffer();

    return new Response(audioBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'audio/mpeg',
        'Cache-Control': 'public, max-age=86400, s-maxage=86400',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
      },
    });
  } catch (err) {
    return new Response(`Error: ${err.message}`, { status: 500 });
  }
}
