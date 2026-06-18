# Current Action Items: June 15, 2026

Canonical detail lives in `meeting_notes_2026-06-15.md`.

1. **Run the full experiment pipeline.**
   - Diagnose the cluster errors first.
   - Then run all datasets, all models, and all prompt families.
   - Include the rephrasing experiments.

2. **Rerun the pruning experiment with Hadas's code.**
   - Integrate the code she sends into this codebase.
   - Preserve good behavior and test generalization.
   - Be careful about small implementation details, since the results may be sensitive to exact code behavior.

3. **Study the possible connection to in-context learning.**
   - Work through the idea that the model may override knowledge because the prompt context teaches or pressures it to do so.
   - Read more in-context learning and mechanistic interpretability literature with this analogy in mind.

4. **Read the papers more carefully.**
   - Revisit the collected papers and extract the key claims, methods, assumptions, and caveats relevant to the project.
