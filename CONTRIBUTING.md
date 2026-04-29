# Contributing to Embedded Systems & AI

Thank you for contributing. This book is a working draft and all chapters are open for review, content additions, corrections, and examples.

---

## How to Contribute

### 1. Find the chapter you want to edit

All chapters are in the `chapters/` folder. Each file is named clearly:

```
chapters/
  chapter-01-when-ai-meets-constrained-hardware.md
  chapter-02-embedded-constraints-as-design-variables.md
  ...
  chapter-14-integration-case-studies.md
  appendix-a-embedded-quick-reference.md
  ...
```

### 2. Create a branch named after the chapter

Use this naming convention:

```
chapter-01/your-description
chapter-05/add-activation-memory-example
chapter-10/fix-wcet-section
appendix-c/update-hardware-table
```

```bash
git checkout -b chapter-05/add-activation-memory-example
```

### 3. Make your edits

- Fill in `<!-- TODO -->` sections with content
- Fix errors in existing content
- Add worked examples, diagrams (as images in `assets/`), or tables
- Keep the structure (headings, learning outcomes) intact — ask before restructuring

### 4. Open a Pull Request

- Target the `main` branch
- Title: `[Ch X] Short description` e.g. `[Ch 5] Add activation memory buffer reuse example`
- In the description, briefly explain what you added/changed and why

The author will review, leave comments, and merge when ready.

---

## What Needs Work

Sections marked `<!-- TODO -->` are open for contributions. Priority areas:

| Chapter | What's needed |
|---------|--------------|
| Ch 1 | Opening failure case story |
| Ch 3 | ML translation table expansion |
| Ch 4 | Keyword spotting worked example (Arduino) |
| Ch 7 | Battery life worked example (camera trap) |
| Ch 10 | WCET analysis section — needs SME review |
| Ch 14 | All three case studies |
| Appendix D | Edge Impulse and STM32Cube.AI setup steps |

---

## Style Guide

- Write for **final-year undergraduates** with embedded background, no ML background
- Use concrete numbers over vague descriptions ("256 KB SRAM" not "limited memory")
- Every claim about performance should cite a source or be reproducible from the worked example
- Use `<!-- TODO -->` for placeholders, not empty sections
- Tables are preferred over bullet lists for comparisons

---

## Questions?

Open a GitHub Issue with the label `question`.
