"""Render current prediction figures/tables from freshly computed reports."""
import json
from pathlib import Path

import numpy as np

NAMES = ["Muse Spark~1.2", "GPT-5.4", "GPT-5.5", "Claude Opus~4.8"]


def combined_comparison_table(data, n):
    from scripts.report_joint_allocation import METHODS
    if n == 2 and all(f'{m}-n4' in data['reports'] for m in ('muse', 'gpt55')):
        lines = [r'\begin{table}[htbp]', r'\centering\small\setlength{\tabcolsep}{4pt}',
                 r'\begin{tabular}{lrrrrr}', r'\toprule Framework & Muse & GPT-5.4 & GPT-5.5 & Opus & Average \\']
        for allocation, models in [(2, ['muse', 'gpt54', 'gpt55', 'opus']), (4, ['muse', 'gpt55'])]:
            lines += [r'\midrule', r'\multicolumn{6}{l}{\textit{$N='+str(allocation)+r'$}} \\', r'\midrule']
            assert all(data['reports'][f'{m}-n{allocation}']['trials'] == 171 for m in models)
            values = {m: [data['reports'][f'{m}-n{allocation}']['metrics'][method]['rmse'] for method in METHODS] for m in models}
            values['pooled'] = np.sqrt(np.mean(np.square([values[m] for m in models]), axis=0)).tolist()
            for index, label in enumerate(METHODS.values()):
                if label == 'DE':
                    lines.append(r'\midrule')
                if label in ('DE', 'R-DE'):
                    label += ' (ours)'
                cells = []
                for model in ['muse', 'gpt54', 'gpt55', 'opus', 'pooled']:
                    if model not in values:
                        cells.append('---')
                    else:
                        value = values[model][index]
                        cells.append(r'\textbf{'+f'{value:.2f}'+'}' if value == min(values[model]) else f'{value:.2f}')
                lines.append(label+' & '+' & '.join(cells)+r' \\')
        lines += [r'\bottomrule', r'\end{tabular}',
                  r'\caption{Aggregate-curve RMSE in solved-trial counts for $N=2$ and $N=4$; each reported model has 57 problems and 171 allocation trials. A trial succeeds if any of its $N$ trajectories solves. The Average is the square root of the mean model-specific MSE (four models for $N=2$, two for $N=4$). Lower is better; bold marks column minima within each panel. Dashes indicate unavailable $N=4$ coverage.}',
                  r'\label{tab:n2-predictor-comparison}', r'\end{table}']
        return '\n'.join(lines)+'\n'
    models = ['muse', 'gpt54', 'gpt55', 'opus']
    names = ['Muse', 'GPT-5.4', 'GPT-5.5', 'Opus']
    metrics = data['reports']
    assert all(metrics[f'{m}-n{n}']['trials'] == 171 for m in models)
    individual = np.array([[metrics[f'{m}-n{n}']['metrics'][method]['rmse'] for m in models] for method in METHODS])
    values = np.column_stack([individual, np.sqrt(np.mean(individual**2, axis=1))])
    minima = values.min(axis=0)
    lines = [r'\begin{table}[htbp]', r'\centering\small\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{l'+'r'*(len(models)+1)+'}',
             r'\toprule Framework & '+' & '.join(names)+r' & Average \\', r'\midrule']
    for (method, label), row in zip(METHODS.items(), values):
        if label == 'DE':
            lines.append(r'\midrule')
        if label in ('DE', 'R-DE'):
            label += ' (ours)'
        cells = [r'\textbf{'+f'{v:.2f}'+'}' if v == lo else f'{v:.2f}' for v, lo in zip(row, minima)]
        lines.append(label+' & '+' & '.join(cells)+r' \\')
    label = 'predictor-comparison' if n == 1 else 'n2-predictor-comparison'
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{Full-curve $N='+str(n)+r'$ RMSE in solved-trial counts across $K=1,\ldots,'+str(8//n)+r'$; 57 problems and 171 '+('trajectories' if n == 1 else 'paired trials')+r' per model. Lower is better.}',
              r'\label{tab:'+label+'}', r'\end{table}']
    return '\n'.join(lines)+'\n'


def comparison_table(data, n):
    """Render the combined main fit, or explicitly marked transfer experiments."""
    if all(fit.get('prior_dataset') is None for fit in data.get('fits', {}).values()):
        return combined_comparison_table(data, n)
    from scripts.report_joint_allocation import METHODS, summarize
    models = ['muse', 'gpt54', 'gpt55', 'opus']
    lines = [r'\begin{table}[htbp]', r'\centering\small\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{lrrrrr}',
             r'\toprule Framework & Muse & GPT-5.4 & GPT-5.5 & Opus & Average \\']
    for allocation in ([1] if n == 1 else [2, 4]):
        for dataset, name, expected in [('results', 'AOBench', 35), ('results-imobench', 'IMO-ProofBench', 22)]:
            title = name + (r' ($N='+str(allocation)+'$)' if n != 1 else '')
            lines += [r'\midrule', r'\multicolumn{6}{l}{\textit{'+title+r'}} \\', r'\midrule']
            values = {}
            for model in models:
                report = data['reports'].get(f'{model}-n{allocation}')
                if report is None:
                    continue
                subset = [r for r in report['rows'] if r['dataset'] == dataset]
                if len(subset) != expected or sum(r['weight'] for r in subset) != 3*expected:
                    continue
                metrics = summarize(subset)['metrics']
                values[model] = [metrics[method]['rmse'] for method in METHODS]
            values['average'] = np.sqrt(np.mean(np.square(list(values.values())), axis=0)).tolist()
            for i, label in enumerate(METHODS.values()):
                if label == 'DE':
                    lines.append(r'\midrule')
                if label in ('DE', 'R-DE'):
                    label += ' (ours)'
                cells = []
                for model in models+['average']:
                    if model not in values:
                        cells.append('---')
                    else:
                        value=values[model][i]
                        cells.append(r'\textbf{'+f'{value:.2f}'+'}' if value==min(values[model]) else f'{value:.2f}')
                lines.append(label+' & '+' & '.join(cells)+r' \\')
    caption = (r'Full-curve RMSE in solved-trial counts for $N=1$ across eight checkpoints.' if n == 1 else
               r'Aggregate-curve RMSE in solved-trial counts for $N=2$ and $N=4$. A trial succeeds if any constituent trajectory solves.')
    caption += r' AOBench contains 35 problems and 105 trials; IMO-ProofBench contains 22 problems and 66 trials. R-DE priors are fitted only on AOBench and frozen for IMO-ProofBench; both datasets provide per-problem intervention measurements. Lower is better; compare frameworks within each panel.'
    if n != 1:
        caption += r' Dashes indicate unavailable complete cohorts; each Average uses the available models in that panel.'
    label='predictor-comparison' if n==1 else 'n2-predictor-comparison'
    lines += [r'\bottomrule', r'\end{tabular}', r'\caption{'+caption+'}', r'\label{tab:'+label+'}', r'\end{table}']
    return '\n'.join(lines)+'\n'


def execution_subgroup_table(data):
    """Render the oracle-defined execution-nontrivial N=1 sensitivity table."""
    from scripts.report_joint_allocation import METHODS
    models = ['muse', 'gpt54', 'gpt55', 'opus']
    names = ['Muse', 'GPT-5.4', 'GPT-5.5', 'Opus']
    counts = []
    values = []
    for method in METHODS:
        row = []
        for model in models:
            subset = [r for r in data['reports'][f'{model}-n1']['rows'] if r['epsilon'][0] < 1]
            if method == next(iter(METHODS)):
                counts.append(len(subset))
            weights = np.array([r['weight'] for r in subset])
            observed = weights @ np.array([r['observed'] for r in subset])
            predicted = weights @ np.array([r['predictions'][method] for r in subset])
            row.append(float(100 * np.sqrt(np.mean((predicted-observed)**2)) / weights.sum()))
        row.append(float(np.sqrt(np.mean(np.square(row)))))
        values.append(row)
    lines = [r'\begin{table}[t]', r'\centering\small\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{lrrrrr}',
             r'\toprule Framework & '+' & '.join(names)+r' & Average \\', r'\midrule']
    for index, ((_, label), row) in enumerate(zip(METHODS.items(), values)):
        if index == 3:
            lines.append(r'\midrule')
        if label in ('DE', 'R-DE'):
            label += ' (ours)'
        lines.append(label+' & '+' & '.join(f'{value:.2f}' for value in row)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{Full-curve RMSE in percentage points on problems with $\widehat\varepsilon_n(1)<1$: '
              + ', '.join(f'{count} for {name}' for count, name in zip(counts, names))
              + r'. Lower is better.}',
              r'\label{tab:execution-subgroup}', r'\end{table}']
    return '\n'.join(lines)+'\n'


def render(data, output_dir):
    """No fitting or saved analysis dependencies; no unrelated paper assets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'allocation_report.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    from scripts.report_execution_regimes import PANELS, build, render as render_regimes, render_early_gains
    if all(f'{model}-n{n}' in data['reports'] for n, models in PANELS.items() for model in models):
        regimes = build(data)
        (output_dir/'execution_regimes_table.tex').write_text(render_regimes(regimes))
        (output_dir/'early_execution_gains_table.tex').write_text(render_early_gains(regimes))
    for n, filename in [(1, 'allocation_model_fit.tex'), (2, 'allocation_model_replication.tex')]:
        keys = [f'{m}-n{n}' for m in ['muse', 'gpt54', 'gpt55', 'opus']]
        if all(k in data['reports'] for k in keys):
            (output_dir/filename).write_text(plot([data['reports'][k]['rows'] for k in keys], n2=n == 2)+'\n')
        if f'n{n}' in data['pooled_rmse_pp']:
            name = 'predictor_comparison_table.tex' if n == 1 else 'n2_predictor_comparison_table.tex'
            (output_dir/name).write_text(comparison_table(data, n))
    if 'n1' in data['pooled_rmse_pp']:
        lines = [r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{lrrr}',
                 r'\toprule Model & MAE & RMSE & Final observed/predicted \\', r'\midrule',
                 r'\multicolumn{4}{l}{\textit{$N=1,K=8$: 171 trajectories per model}} \\']
        for key, name in zip(['muse', 'gpt54', 'gpt55', 'opus'], NAMES):
            s = data['reports'][f'{key}-n1']['metrics']['both_regularized']
            lines.append(f"{name} & {s['mae']:.2f} & {s['rmse']:.2f} & ${s['observed'][-1]:.0f}/{s['predicted'][-1]:.1f}$ "+r'\\')
        lines += [r'\bottomrule', r'\end{tabular}', r'\caption{R-DE single-trajectory errors in passing-trajectory counts, with 171 trajectories per model.}', r'\label{tab:allocation-model-error}', r'\end{table}']
        (output_dir/'joint_error_table.tex').write_text('\n'.join(lines)+'\n')
        (output_dir/'execution_subgroup_table.tex').write_text(execution_subgroup_table(data))


def coords(x,y):
    return 'coordinates {'+' '.join(f'({a},{b:.6f})' for a,b in zip(x,y))+'};'


def plot(rows_by_model,n2=False,baseline=False):
    lines=[r'\begingroup',r'\definecolor{fitblue}{HTML}{356AA0}',r'\definecolor{fitorange}{HTML}{D97706}',r'\definecolor{fitpurple}{HTML}{7A5195}',r'\begin{tikzpicture}[font=\sffamily\small]',r'\begin{groupplot}[group style={group size=4 by 1,horizontal sep=0.38cm},scale only axis,width=0.204\linewidth,height=2.5cm,xmin=0.7,xmax=8.3,ymin=0,ymax=171,xtick={2,4,6,8},xticklabels={$2\times$,$4\times$,$6\times$,$8\times$},ytick={0,50,100,150},tick label style={font=\sffamily\small},axis x line*=bottom,axis y line*=left,tick style={draw=none},axis line style={gray},ymajorgrids,grid style={gray!20},clip=true]']
    for i,rows in enumerate(rows_by_model):
        opts=[]
        if i:opts.append(r'yticklabels=\empty')
        lines.append(r'\nextgroupplot['+','.join(opts)+']')
        K=len(rows[0]['observed']); x=np.arange(1,K+1)*(2 if n2 else 1)
        if baseline:
            for field,color,style in [('observed','black','only marks,mark=*,mark size=1.4pt'),('solved_geometric','orange','dashed'),('oracle_slope_linear','cyan!80!blue','dashed'),('oracle_gain_transfer','red','dashed'),('neither_regularized','green!65!black','dashed'),('both_regularized','magenta','dashed')]:
                y=sum(r['weight']*np.array(r['observed'] if field=='observed' else r['predictions'][field]) for r in rows)
                lines.append(r'\addplot['+f'draw={color},{style},line width=1.3pt,opacity=0.85'+'] '+coords(x,y))
        else:
            groups=[('results','fitblue'),('results-imobench','fitorange'),(None,'fitpurple')]
            for dataset,color in groups:
                subset=[r for r in rows if dataset is None or r['dataset']==dataset]
                if not subset:continue
                obs=sum(r['weight']*np.array(r['observed']) for r in subset)
                pred=sum(r['weight']*np.array(r['predictions']['both_regularized']) for r in subset)
                lines.append(r'\addplot['+f'draw={color},solid,line width=1.5pt,opacity=0.78,mark=*,mark size=1.3pt'+'] '+coords(x,obs))
                lines.append(r'\addplot['+f'draw={color},dashed,line width=1.5pt,opacity=0.78,mark=o,mark size=1.3pt'+'] '+coords(x,pred))
    lines.append(r'\end{groupplot}')
    for i,name in enumerate(NAMES,1):
        lines.append(r'\node[anchor=north,font=\sffamily\small] at ([yshift=-4mm]group c'+str(i)+r'r1.south) {'+name+'};')
    lines.append(r'\node[rotate=90,anchor=south] at ([xshift=-7mm]group c1r1.west) {Cumulative solved};')
    lines.append(r'\node[anchor=north] at ([yshift=-9mm]{$(group c1r1.south)!0.5!(group c4r1.south)$}) {Total inference budget};')
    if baseline:
        legend=[(-50,11,'black','only marks','Observed'),(-17,11,'orange','dashed','Plain-geometric'),(30,11,'cyan!80!blue','dashed','Linear'),(-50,5,'red','dashed','OGT'),(-17,5,'green!65!black','dashed','DE'),(30,5,'magenta','dashed','R-DE')]
    else:legend=[(-48,5,'fitblue','solid','AOBench'),(-18,5,'fitorange','solid','IMO-ProofBench'),(27,5,'fitpurple','solid','Combined')]
    for offset,yshift,color,style,name in legend:
        anchor=r'{$(group c1r1.north)!0.5!(group c4r1.north)$}'
        if style == 'only marks':
            lines.append(r'\fill['+color+r'] ([xshift='+str(offset+2.5)+r'mm,yshift='+str(yshift)+r'mm]'+anchor+r') circle[radius=1.4pt];')
        else:
            lines.append(r'\draw['+color+','+style+r',line width=1.5pt] ([xshift='+str(offset)+r'mm,yshift='+str(yshift)+r'mm]'+anchor+r') -- ++(5mm,0);')
        lines.append(r'\node[anchor=west,draw=none,fill=none,inner sep=0pt] at ([xshift='+str(offset+7)+r'mm,yshift='+str(yshift)+r'mm]'+anchor+r') {'+name+'};')
    lines += [r'\end{tikzpicture}',r'\endgroup']
    return '\n'.join(lines)
