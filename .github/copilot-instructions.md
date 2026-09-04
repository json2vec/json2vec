# GitHub Copilot Instructions

Read `AGENTS.md` first and follow its implementation style and ownership
boundaries. Use `CONTRIBUTING.md` for the complete extension, Arrow data,
documentation, and testing contracts. Those files are authoritative if this
summary differs.

Use `import relflow as rf` in generated examples. Build public schemas with
`rf.Model(...)`, `rf.Branch(...)`, and top-level tensorfield constructors. Do
not invent a public `Struct(...)` API.

Keep code direct: use short semantic names, inline one-use forwarding helpers,
never prefix functions or classes with `_`, preserve Arrow as the canonical CPU
representation, and keep datatype behavior inside its extension. Keep examples
runnable and update tests and docs with public behavior.
