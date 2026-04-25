# MCP Protocol Reference

## Connection lifecycle

Every MCP session has three phases:

**1. Initialization**: Client sends `initialize` with protocol version + capabilities → Server responds with its capabilities → Client sends `initialized` notification. Neither party sends non-ping requests during this phase.

**2. Operation**: Normal request/response/notification exchange based on negotiated capabilities.

**3. Shutdown**: STDIO → close stdin, wait for exit, SIGTERM if needed. HTTP → HTTP DELETE with `Mcp-Session-Id` header.

Protocol versions (date-based): `2024-11-05`, `2025-03-26`, `2025-06-18`, `2025-11-25`.

## STDIO transport

- Server runs as **subprocess** of client
- Reads JSON-RPC from **stdin**, writes to **stdout**
- Messages are **newline-delimited**, no embedded newlines
- Logging goes to **stderr** only — never write non-MCP to stdout
- Single client per server (1:1)
- Best for: local tools, desktop integrations, CLIs

## Streamable HTTP transport

Replaced deprecated SSE transport (spec 2025-03-26). Single endpoint (default `/mcp`).

- Client → Server: always HTTP POST
- Notifications: server returns `202 Accepted`
- Requests: server returns `application/json` (simple) or `text/event-stream` (SSE with possible server-initiated messages)
- Session management: `Mcp-Session-Id` header
- Resumability via SSE event IDs
- Clients may open long-lived GET for server-initiated messages
- Shutdown: HTTP DELETE with session ID header

### Stateless mode (serverless)

```python
mcp = FastMCP("Serverless", stateless_http=True)
# Each request creates a fresh session — works with Lambda, Cloud Run, etc.
```

## Roots — filesystem boundaries

Roots are URIs the client shares with the server to indicate its operating scope. Informational, not enforced.

- Clients declare: `"roots": {"listChanged": true}` in capabilities
- Servers discover via `roots/list` request
- Changes notified via `notifications/roots/list_changed`
- Must be `file://` URIs
- Servers SHOULD validate path operations against roots

```json
{"roots": [
  {"uri": "file:///home/user/project", "name": "My Project"}
]}
```

## Security model

Three trust boundaries: host, client, server. Key principles:

- Users must consent to all data access
- Servers are treated as **potentially untrusted**
- Tool annotations are **advisory hints, not security controls**
- Hosts must obtain permission before exposing user data

### Known threats

| Threat | Description | Mitigation |
|---|---|---|
| Tool poisoning | Hidden malicious instructions in tool descriptions | Review tool descriptions, deny-by-default |
| Prompt injection | Hostile content in tool output treated as trusted | Validate at every trust boundary |
| Confused deputy | Proxy servers exploited via token reuse | Audience-bound, short-lived tokens |
| Rug-pull attacks | Tools change behavior after approval | Monitor tool description changes, reset on `tools/list_changed` |
| SSRF | Server induces client to fetch internal resources | Validate URLs, restrict network access |
| Session hijacking | Attacker captures session ID | Use TLS, rotate session IDs |

### Approval workflow best practices

| Tool risk level | Recommended flow |
|---|---|
| Read-only (`readOnlyHint: true`) | Auto-approve |
| Atomic writes | Single confirmation |
| Impactful operations | Confirmation + audit log |
| Destructive (`destructiveHint: true`) | Multi-party approval |

### Audit logging

Log every tool invocation: timestamp, tool name, parameters, user identity, result. Correlate with request IDs across multi-server workflows. MCP's native `notifications/message` logging is server-to-client only. For end-to-end tracing, use gateway solutions.

## OAuth 2.1 (required for HTTP remote servers)

Flow:
1. Client connects unauthenticated → `401 Unauthorized` with `WWW-Authenticate`
2. Client fetches Protected Resource Metadata from `/.well-known/oauth-protected-resource`
3. Client fetches Authorization Server Metadata from `/.well-known/oauth-authorization-server`
4. Dynamic Client Registration (no manual app registration)
5. Authorization Code Grant with PKCE (`S256` mandatory)
6. Client uses `Authorization: Bearer <token>` on subsequent requests

Token validation: check issuer, audience, expiry, scopes. Use short-lived tokens with refresh rotation.

## Client-side permission policies

**No standard `permissions.json` format exists.** Each client implements its own:

- **VS Code / Copilot**: Tools disabled by default, user enables per-tool. Resets on `tools/list_changed`.
- **Amazon Q Developer**: Registry JSON allowlists fetched over HTTPS.
- **LiteLLM**: YAML per-entity access controls.
- **Gateways (Permit.io, Peta)**: Centralized dashboards.

### Server connection config (`mcp.json` / `.mcp.json`)

Most clients use this format for server registration:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/dir"]
    },
    "remote": {
      "url": "https://api.example.com/mcp"
    }
  }
}
```

## Progress notifications

Requester includes `progressToken` in request metadata. Handler sends `notifications/progress`.

```python
# Server-side (FastMCP handles token check automatically)
@mcp.tool()
async def long_task(items: list[str], ctx: Context) -> str:
    for i, item in enumerate(items):
        await ctx.report_progress(i + 1, len(items), f"Processing {item}")
        await process(item)
    return "Done"
```

## Cancellation

Either side sends `notifications/cancelled` with `requestId` and optional `reason`. Handle race conditions — cancellation may arrive after completion.

## Ping / keepalive

Simple `{"method": "ping"}` → `{"result": {}}`. Either side can initiate. Receiver must respond promptly.

## Pagination

All list operations use cursor-based pagination. Server includes `nextCursor` in response; client passes it in next request. Page size is server-determined.

## Resource subscriptions

Clients subscribe via `resources/subscribe`. Server sends `notifications/resources/updated` on changes. Requires `"resources": {"subscribe": true}` in server capabilities.

## Completion / autocomplete

`completion/complete` provides suggestions for prompt arguments and resource template parameters. Max 100 values per response. Supports contextual completions via `context.arguments`.

## Logging levels

Eight levels (debug → emergency) via `notifications/message`. Client controls minimum level with `logging/setLevel`. FastMCP: `await ctx.info("msg")`, `await ctx.debug("msg")`, etc.

## Server discovery (evolving)

No finalized standard yet. Active proposals:
- **SEP-1649**: `/.well-known/mcp.json` server cards
- **SEP-1960**: `/.well-known/mcp` endpoint
- **IETF draft**: `mcp://` URI scheme with DNS TXT fallback

Today: manual `mcp.json` configuration.

## CLI tools

```bash
mcp dev server.py                    # MCP Inspector (debug UI)
mcp dev server.py --with pandas      # With extra deps
mcp install server.py                # Register in Claude Desktop
mcp install server.py -e API_KEY=x   # With env vars
mcp run server.py                    # Run directly
```
