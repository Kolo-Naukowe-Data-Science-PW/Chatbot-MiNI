# MiNIonek Frontend

React + Vite chat UI for the MiNIonek chatbot.

## Features

- Chat interface with conversation history
- Role selection (student / staff / other) — affects LLM prompt hints
- A/B testing support (random model variant assignment per session)
- Thumbs up/down feedback sent to `/feedback` endpoint
- Free-text comment on any answer, sent to the existing `/errors` endpoint
- Source URL display under each answer
- Full interface in PL / EN / UA (dictionary in `src/i18n.js`); the major and year are stored as keys and always sent to the backend in Polish
- Mobile layout (≤ 768 px): the sidebar becomes a slide-out panel
- Question length limit of 1000 characters (`src/queryLimit.js`, mirrors `QUERY_MAX_CHARS` in the API) and a localised message on HTTP 429

## Dev setup

```bash
npm install
npm run dev       # http://localhost:5173 — proxies /api/* to localhost:8000
```

## Production (Docker)

```bash
npm run build
npm run preview   # proxies /api/* to http://api:8000 (Docker internal DNS)
```

The proxy is configured in `vite.config.js` for both `server` (dev) and `preview` (production) modes.

## API endpoints used

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/chat` | POST | Send message, receive answer + sources |
| `/api/feedback` | POST | Send thumbs up/down rating |
