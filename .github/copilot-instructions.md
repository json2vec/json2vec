# GitHub Copilot Instructions

Read `AGENTS.md` first. It defines the public JSON2Vec API style, schema patterns, gotchas, and verification commands.

Use `import json2vec as j2v` in generated examples. Prefer `Model.from_schema(...)`, built-in tensorfield constructors, `Array(...)`, and top-level package exports. Do not invent a public `Struct(...)` API.

Keep inline docs examples and Quarto pages runnable and small. When changing package behavior, update tests and docs together.
