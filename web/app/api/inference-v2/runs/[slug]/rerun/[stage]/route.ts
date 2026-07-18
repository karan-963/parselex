import { NextResponse } from 'next/server';

const FASTAPI_URL = process.env.FASTAPI_URL ?? 'http://localhost:8000';

type RouteCtx = { params: Promise<{ slug: string; stage: string }> };

export async function POST(req: Request, { params }: RouteCtx) {
  const { slug, stage } = await params;
  try {
    const precision = new URL(req.url).searchParams.get('precision') ?? 'fp32';
    const res = await fetch(
      `${FASTAPI_URL}/inference-v2/runs/${encodeURIComponent(slug)}/rerun/${encodeURIComponent(stage)}?precision=${encodeURIComponent(precision)}`,
      {
        method: 'POST',
      }
    );
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 502 });
  }
}
