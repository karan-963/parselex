import { NextResponse } from 'next/server';

const FASTAPI_URL = process.env.FASTAPI_URL ?? 'http://localhost:8000';

type RouteCtx = { params: Promise<{ slug: string }> };

export async function GET(_req: Request, { params }: RouteCtx) {
  const { slug } = await params;
  try {
    const res = await fetch(`${FASTAPI_URL}/inference-v2/runs/${encodeURIComponent(slug)}/pdf`);
    if (!res.ok) {
      return NextResponse.json({ error: 'PDF not found' }, { status: res.status });
    }
    const blob = await res.blob();
    return new NextResponse(blob, {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': `inline; filename="${slug}.pdf"`,
      },
    });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 502 });
  }
}
