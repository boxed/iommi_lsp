# History

## Unreleased

- Another fallback to handle `instance.id` and similar django-isms
- Model FK assignment no longer warns about mismatched types
- `annotate()` support
- Reverse mapping issue fixed
- Subquery analysis fixed

## 0.0.3 (2026-05-16)

- Big optimization for full-file analysis, and another massive optimization on top
- FK improvements
- iommi `Table.header` improvements
- More django-isms handled
- Reverse relationships fix
- Reverse relation support
- F-object comparison ops support
- `Model.pk` special case
- Comprehension support
- `ForeignKey` subclasses handled correctly
- iommi `attrs` namespace fix
- Primary key member handling for Django models fixed
- Redirect to `.` and `..` no longer warns
- `extra` / `extra_evaluated` support
- iommi graph rebuilds lazily on LSP startup — faster startup, self-healing when the graph is stale
- iommi `@refinable` members fixed
- Tests fixed on GitHub Actions
- Avoid an incorrect `IntegerChoices` warning
- Fix for a `Column` `cell__attrs__class` false warning
- iommi completion fixes

## 0.0.2 (2026-05-15)

- Template completions, and more
- Lots of improved completions; some performance fixes
- Autocomplete for `foo.bar_id`
- Suppress `model_instance.foreign_key_id` warnings from ty
- Perf fixes
- Fixed `INSTALLED_APPS` completion
- Completion fix for bug where a previous `"""` anywhere in the file would stop completion from happening
- Template filename completion works
- iommi `class Meta` support
- Suppress "unused" `request` parameter for Django views
- Consistent use of `iommi_lsp` over `iommi-lsp`
- Ordering
- Bundle ty; improved completion
- iommi completions improvements and fixes
- More ORM completion magic
- More lookups
- Inspects work much better

## 0.0.1 (2026-05-10)

- Initial release
- Basic Django support
- iommi support
- Django-specific patterns: `order_by`, `values`, `F`, and more
- `filters`, iommi, `Q` objects
