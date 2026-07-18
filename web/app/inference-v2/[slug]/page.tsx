import InferenceV2ResultClient from './InferenceV2ResultClient';

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  return { title: `Inference V2 — ${decodeURIComponent(slug)}` };
}

export default async function InferenceV2ResultPage({ params }: PageProps) {
  const { slug } = await params;
  return <InferenceV2ResultClient slug={decodeURIComponent(slug)} />;
}
