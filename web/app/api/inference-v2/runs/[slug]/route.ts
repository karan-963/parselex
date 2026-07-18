import { NextResponse } from 'next/server';

const FASTAPI_URL = process.env.FASTAPI_URL ?? 'http://localhost:8000';

type RouteCtx = { params: Promise<{ slug: string }> };

export async function GET(_req: Request, { params }: RouteCtx) {
  const { slug } = await params;
  try {
    const res = await fetch(`${FASTAPI_URL}/inference-v2/runs/${encodeURIComponent(slug)}`, {
      cache: 'no-store',
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 502 });
  }
}
