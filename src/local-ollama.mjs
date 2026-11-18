const LOOPBACK_IPV4 = /^127(?:\.\d{1,3}){3}$/;

export function validateOllamaBaseUrl(baseUrl) {
  let parsed;
  try {
    parsed = new URL(baseUrl);
  } catch {
    throw new Error('OLLAMA_HOST must be an HTTP loopback origin');
  }

  const hostname = parsed.hostname.toLowerCase();
  const isLoopback = hostname === 'localhost' || hostname === '[::1]' || LOOPBACK_IPV4.test(hostname);
  if (
    parsed.protocol !== 'http:'
    || !isLoopback
    || parsed.username
    || parsed.password
    || !['', '/'].includes(parsed.pathname)
    || parsed.search
    || parsed.hash
  ) {
    throw new Error('OLLAMA_HOST must be an HTTP loopback origin');
  }
  return parsed.origin;
}
