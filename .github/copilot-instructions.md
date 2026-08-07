# GitHub Copilot Instructions

Read `AGENTS.md` first. It defines the public RelFlow API style, schema patterns, gotchas, and verification commands.

Use `import relflow as rf` in generated examples. Prefer `Model.from_tree(...)`, built-in tensorfield constructors, `Branch(...)`, and top-level package exports. Do not invent a public `Struct(...)` API.

Keep inline docs examples and Quarto pages runnable and small. When changing package behavior, update tests and docs together.
