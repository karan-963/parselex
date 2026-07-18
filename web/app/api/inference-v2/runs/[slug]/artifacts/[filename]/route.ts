import { NextResponse } from 'next/server';

const FASTAPI_URL = process.env.FASTAPI_URL ?? 'http://localhost:8000';

type RouteCtx = { params: Promise<{ slug: string; filename: string }> };

export async function GET(_req: Request, { params }: RouteCtx) {
  const { slug, filename } = await params;
  try {
    const res = await fetch(
      `${FASTAPI_URL}/inference-v2/runs/${encodeURIComponent(slug)}/artifacts/${encodeURIComponent(filename)}`,
    );
    if (!res.ok) {
      return NextResponse.json({ error: 'Artifact not found' }, { status: res.status });
    }
    const blob = await res.blob();
    const contentType = filename.endsWith('.json') ? 'application/json' : 'application/octet-stream';
    return new NextResponse(blob, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="${filename}"`,
      },
    });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 502 });
  }
}
