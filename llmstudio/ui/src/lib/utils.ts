import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const generateStream = async (
  url: string,
  data: unknown
): Promise<AsyncIterable<string>> => {
  let response: Response | null = null;
  try {
    response = await fetch(url, {
      method: 'post',
      body: JSON.stringify(data),
      headers: {
        'Content-Type': 'application/json;charset=UTF-8',
      },
    });
    if (!response.ok || !response.body) throw new Error();
    return getIterableStream(response.body);
  } catch (e) {
    throw new Error(getErrorMessage(response?.status));
  }
};

export async function* getIterableStream(
  body: ReadableStream<Uint8Array>
): AsyncIterable<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    const decodedChunk = decoder.decode(value, { stream: true });
    yield decodedChunk;
  }
}

export const getErrorMessage = (error: Number | undefined) => {
  switch (error) {
    case 400:
      return "Oops! That's not a valid request.";
    case 401:
      return 'Invalid API key. Please verify and try again.';
    case 403:
      return 'Access denied. Check API key permissions.';
    case 404:
      return 'Resource unavailable. Check the request and retry.';
    case 429:
      return 'Usage limit exceeded. Try again later.';
    case 500:
      return 'Unexpected server issue. Please try again later.';
    case 503:
    case 529:
      return 'Service is temporarily busy. Please retry later.';
    case undefined:
      return 'LLMstudio Engine is not running';
    default:
      return 'Unknown error';
  }
};
