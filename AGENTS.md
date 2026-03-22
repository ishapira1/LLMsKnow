When asked to generate code for a plot:
- Use `seaborn` and call `seaborn.set_style("white")`.
- Use large titles.
- Use axis-label font sizes of at least `15`.
- Use tick-label font sizes of at least `12`.
- Keep the color scheme consistent and predefined.
- If using two contrasting colors, use `#73b3ab` and `#d4651a`.
- Put the legend box below the plot.

When creating or editing Slurm batch scripts in this repo:
- Include `#SBATCH --mail-type=END,FAIL`.
- Include `#SBATCH --mail-user=itaishapira@g.harvard.edu`.
- Only omit or change those mail settings if the user explicitly asks.
