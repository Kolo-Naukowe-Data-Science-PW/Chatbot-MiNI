// Client-side mirror of the server limit on the length of a question
// (QUERY_MAX_CHARS in the API). Keep both values in sync.
export const MAX_QUERY_CHARS = 1000;

/**
 * Shorten `text` to at most `max` characters.
 *
 * @param {string} text
 * @param {number} [max]
 * @returns {{text: string, truncated: boolean}}
 */
export function limitQuery(text, max = MAX_QUERY_CHARS) {
  if (text.length <= max) {
    return { text, truncated: false };
  }
  return { text: text.slice(0, max), truncated: true };
}
