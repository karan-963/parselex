import { NextResponse } from 'next/server';

const FASTAPI_URL = process.env.FASTAPI_URL ?? 'http://localhost:8000';

export async function POST(req: Request) {
  try {
    const precision = new URL(req.url).searchParams.get('precision') ?? 'fp32';
    const formData = await req.formData();
    const res = await fetch(`${FASTAPI_URL}/inference-v2/parse?precision=${encodeURIComponent(precision)}`, {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 502 });
  }
}
