
import os
import numpy as np
import matplotlib.pyplot as plt



def round_arrays(A,B,C,D):
	return A.round(2), B.round(2), C.round(2), D.round(2)




def get_Names_Labels_Colors(Names_UC):

	Names_MC = ['UMC\n(hard)', 'UMC\n(soft)'] # 2 blocks for MC data
	# Names_UC = ['F', 'M'] # 2 blocks for UC data

	Labels_MC = ['LDA/GW', 'LDA/DB'] # 2 bars in each MC block
	Labels_UC = ['SNF', 'Avg'] # 2 bars in each UC block

	Colors_MC = ['#e65e79', '#79c24f'] # 2 colors in each MC block
	Colors_UC = ['#f79646', '#4bacc6'] # 2 colors in each UC block

	Params1 = [Names_MC, Labels_MC, Colors_MC]
	Params2 = [Names_UC, Labels_UC, Colors_UC]

	return Params1,Params2





def find_ylims(M1,E1,M2,E2):

	M = np.concatenate((M1.reshape(-1,M1.shape[-1]), M2.reshape(-1,M2.shape[-1])), axis=0)
	E = np.concatenate((E1.reshape(-1,E1.shape[-1]), E2.reshape(-1,E2.shape[-1])), axis=0)

	MIN = np.min((M-E), axis=0)
	MAX = np.max((M+E), axis=0)

	MIN = np.round(MIN,-1).astype(int) - 10
	MAX = np.round(MAX,-1).astype(int) + 10

	MIN[MIN<0] = 0
	MAX[MAX>100] = 100

	ylims = [(Min,Max) for Min,Max in zip(MIN,MAX)]
	# input(ylims)

	return ylims





def plot_scores(Params_MC, Params_UC, nt_list):

	PathName = 'SCORES/' + '_'.join(['nt']+list(map(str,nt_list))) + '/'

	if not os.path.exists(PathName):
		os.makedirs(PathName)

	Scores_MC, Errors_MC, Names_MC, Labels_MC, Colors_MC = Params_MC
	Scores_UC, Errors_UC, Names_UC, Labels_UC, Colors_UC = Params_UC

	Scores_MC, Errors_MC, Scores_UC, Errors_UC = round_arrays(Scores_MC, Errors_MC, Scores_UC, Errors_UC)

	if Scores_MC.shape[1] == len(nt_list):
		nt_len = len(nt_list)
	else:
		raise ValueError('Number of nt values does not match for "Scores_MC" and "nt_list"')

	# n_methods_MC (2) :  Hard, Soft
	n_methods_MC = len(Scores_MC)

	# M :  #modality
	M  = len(Scores_UC)

	# 2 ->  LDA/GW, LDA/DB
	n_bars_MC = len(Labels_MC)

	# 2 -> SNF, Avg
	n_bars_UC = len(Labels_UC)

	barWidth = 3.5
	spaceInBars = 1.2
	spaceInBlocks = 9
	space_bar_text = 1.0
	# shiftTextToLeft = 1.9
	shiftTextToLeft = 1.6
	# padding_upper_list = [14,32,32]
	padding_upper_list = [11,18,18]
	score_text_factor = 0.8
	xticklabels_factor = 1
	legend_factor = 1

	dist_btw_2AdjBars = barWidth+spaceInBars
	shift_btw_MC_blocks = n_bars_MC*barWidth + (n_bars_MC-1)*spaceInBars + spaceInBlocks
	shift_btw_UC_blocks = n_bars_UC*barWidth + (n_bars_UC-1)*spaceInBars + spaceInBlocks

	# BarLocs_MC:  shape(4,2)  -->  (GW/DB,  Hard/Soft)
	BarLocs_MC = np.array([[shift_btw_MC_blocks*alig_meth + b*dist_btw_2AdjBars for alig_meth in range(n_methods_MC)] for b in range(n_bars_MC)])
	# BarLocs_UC:  shape(2,2)  -->  (SNF/Avg, F/M)
	BarLocs_UC = np.array([[shift_btw_UC_blocks*m + b*dist_btw_2AdjBars for m in range(M)] for b in range(n_bars_UC)]) + n_methods_MC*shift_btw_MC_blocks

	TickLocs_MC = np.average(BarLocs_MC, axis=0)
	TickLocs_UC = np.average(BarLocs_UC, axis=0)
	All_xTicks = np.concatenate((TickLocs_MC, TickLocs_UC))
	All_xTickLabels = [f'{name}' for name in Names_MC] + [f'{name}' for name in Names_UC]


	Metrics = ['Accuracy', 'Sensitivity', 'Specificity']
	Ylims = find_ylims(Scores_MC, Errors_MC, Scores_UC, Errors_UC)

	# number of subplot rows/columns
	default_ncols = 2
	nrows = int(np.ceil(nt_len/default_ncols))
	ncols = min(default_ncols, nt_len)

	error_kw = dict(ecolor='#303030', lw=1, capsize=7, capthick=1)

	for p, (metric,ylims) in enumerate(zip(Metrics,Ylims)):

		fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*12, nrows*6))
		plt.tight_layout(pad=9)
		#plt.tight_layout(h_pad=9, w_pad=9)
		#plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

		# fig.suptitle('', y=0.98, fontsize=20, fontweight='bold')

		# yticks = list(range(30,90+1, 10))
		yticks = list(range(ylims[0], ylims[1]+1, 10 if (ylims[1]-ylims[0])>50 else 5))
		
		#if yticks[1]-yticks[0] == 5:
		#	yticks = yticks[:-1]

		for i, nt in enumerate(nt_list):

			row, col = divmod(i, ncols)

			if nrows==ncols==1:
				ax = axes
			elif nrows==1 and ncols>1:
				ax = axes[col]
			else:
				ax = axes[row,col]

			title = r'$n_t$ = ' + str(nt)

			ax.grid(True, axis='y', lw=0.5)
			ax.set_axisbelow(True)

			# Plotting MCs
			for meth in range(n_bars_MC):

				bars = Scores_MC[:,i,meth,p]
				errs = Errors_MC[:,i,meth,p]
				label = Labels_MC[meth]
				color = Colors_MC[meth]
				locs = BarLocs_MC[meth]

				ax.bar(locs, bars, yerr=errs if np.any(errs) else None, color=color, error_kw=error_kw, width=barWidth, edgecolor='white', label=label)

				for loc, bar, err in zip(locs, bars, errs):
					ax.text(x=loc-shiftTextToLeft, y=bar+err+space_bar_text, s=str(bar), fontsize=14*score_text_factor)


			# Plotting UCs
			for meth in range(n_bars_UC):

				bars = Scores_UC[:,meth,p]
				errs = Errors_UC[:,meth,p]
				label = Labels_UC[meth]
				color = Colors_UC[meth]
				locs = BarLocs_UC[meth]

				ax.bar(locs, bars, yerr=errs if np.any(errs) else None, color=color, error_kw=error_kw, width=barWidth, edgecolor='white', label=label)

				for loc, bar, err in zip(locs, bars, errs):
					ax.text(x=loc-shiftTextToLeft, y=bar+err+space_bar_text, s=str(bar), fontsize=14*score_text_factor)


			ax.set_title(title, pad=15, fontsize=30)
			ax.set_xticks(All_xTicks, minor=False)
			ax.set_xticklabels(All_xTickLabels, fontsize=20*xticklabels_factor)
			ax.set_yticks(yticks, minor=False)
			ax.set_ylim(yticks[0], yticks[-1]+padding_upper_list[p])
			ax.set_ylabel(f'{metric} (%)', labelpad=10, fontsize=18)
			ax.legend(loc='upper left', fontsize=10*legend_factor)
			# ax.legend(loc='upper right', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5), fontsize=10*legend_factor)

		plt.savefig(PathName+f'{metric[:3] if p==0 else metric[:4]}.png', dpi=200)
		fig.clf()