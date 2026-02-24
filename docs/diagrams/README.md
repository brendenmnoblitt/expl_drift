# Diagram Docs

This directory is organized with one folder per diagram.

## Layout

- `architecture_pipeline/`: Pipeline overview diagram and explainer.
- `alert_state_machine/`: Alert state machine diagram and explainer.
- `class_conditional_flow/`: Class-conditional flow diagram and explainer.

Each folder contains:

- `diagram.mmd`: Mermaid source.
- `README.md`: Explanation of what the diagram represents and how to read it.

Experiment-specific diagrams remain in:
`../expl_drift_experiments/docs/diagrams/`

## Export to SVG or PNG

If Mermaid CLI is installed (`mmdc`):

```bash
mmdc -i docs/diagrams/architecture_pipeline/diagram.mmd -o docs/diagrams/architecture_pipeline/diagram.svg
mmdc -i docs/diagrams/alert_state_machine/diagram.mmd -o docs/diagrams/alert_state_machine/diagram.svg
mmdc -i docs/diagrams/class_conditional_flow/diagram.mmd -o docs/diagrams/class_conditional_flow/diagram.svg
```

Or with Docker:

```bash
docker run --rm -u "$(id -u):$(id -g)" \
  -v "$PWD:/data" minlag/mermaid-cli \
  -i /data/docs/diagrams/architecture_pipeline/diagram.mmd \
  -o /data/docs/diagrams/architecture_pipeline/diagram.svg
```
