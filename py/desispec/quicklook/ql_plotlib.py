"""
desispec.quicklook.ql_plotlib
=============================

Generic plotting algorithms for QuickLook QAs.
"""
import numpy as np
import matplotlib.pyplot as plt

def ql_qaplot(fig,plotconf,qadict,camera,expid,outfile):
    """
    Get plotting configuration info and setup plots

    Args:
        fig: matplotlib figure
        plotconf: list of config info for each plot
        qadict: QA metrics dictionary
        camera, expid: to be used in output png title
        outfile: output png file

    Returns:
        png file containing all desired plots
    """
    #- Find relevant plots in plotting configuration file
    plotconfig=[]
    for page in plotconf:
        for plot in plotconf[page]:
            if plot != 'Title':
                for key in plotconf[page][plot]:
                    met=None
                    if key == 'VALS' and plotconf[page][plot]['TYPE'] == 'PATCH':
                        met=str(plotconf[page][plot][key])
                    elif key == 'YVALS' and plotconf[page][plot]['TYPE'] == '2DPLOT':
                        met=str(plotconf[page][plot][key])
                    elif key == 'ZVALS' and plotconf[page][plot]['TYPE'] == '3DPLOT':
                        met=str(plotconf[page][plot][key])
                    if met and met in qadict["METRICS"]:
                        title=plotconf[page]['Title']
                        plotconfig.append(plotconf[page][plot])

    hardplots=False
    if len(plotconfig) != 0:
        #- Setup patch plot
        plt.suptitle("{}, Camera: {}, Expid: {}".format(title,camera,expid),fontsize=10)

        #- Loop through all plots in configuration file
        nplots=len(plotconfig)
        nrow=ncol=int(np.ceil(np.sqrt(len(plotconfig))))
        for p in range(nplots):
            #- Grab necessary plot config info
            plot=plotconfig[p]
            plottype=plot['TYPE']
            plottitle=plot['PLOT_TITLE']
            #- Optional plot config inputs
            heatmap=None
            if 'HEAT' in plot:
                heatmap=plot['HEAT']
            xlim=None
            if 'XRANGE' in plot:
                xlim=plot['XRANGE']
            ylim=None
            if 'YRANGE' in plot:
                ylim=plot['YRANGE']
            zlim=None
            if 'ZRANGE' in plot:
                zlim=plot['ZRANGE']

            #- Generate subplots
            ax=fig.add_subplot('{}{}{}'.format(nrow,ncol,p+1))
            if plottype == 'PATCH':
                vals=np.array(qadict['METRICS'][plot['VALS']])
                grid=plot['GRID']
                patch=ql_patchplot(ax,vals,plottitle,grid,heatmap)
                fig.colorbar(patch)
            if plottype == '2DPLOT':
                xvals=np.array(qadict['METRICS'][plot['XVALS']])
                yvals=np.array(qadict['METRICS'][plot['YVALS']])
                xtitle=plot['XTITLE']
                ytitle=plot['YTITLE']
                ql_2dplot(ax,xvals,yvals,plottitle,xtitle,ytitle,xlim,ylim)
            if plottype == '3DPLOT':
                xvals=np.array(qadict['METRICS'][plot['XVALS']])
                yvals=np.array(qadict['METRICS'][plot['YVALS']])
                zvals=np.array(qadict['METRICS'][plot['ZVALS']])
                xtitle=plot['XTITLE']
                ytitle=plot['YTITLE']
                scatter=ql_3dplot(ax,xvals,yvals,zvals,plottitle,xtitle,ytitle,zlim,heatmap)
                fig.colorbar(scatter)

        #- Adjust plots to fit page and output png
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        fig.savefig(outfile)

    #- If QA not in plot config, use hard coded plots
    else:
        hardplots=True

    return hardplots

def ql_patchplot(ax,vals,plottitle,grid,heatmap=None):
    """
    Make patch plot of specific metrics provided in configuration file

    Args:
        ax: matplotlib subplot
        vals: QA metric to be plotted
        plottitle: plot title from configuration file
        grid: shape of patch plot
    Optional:
        heat: specify color of heatmap (must conform to matplotlib)

    Returns:
        matplotlib sublot containing plotted metrics
    """
    #- Setup title and tick parameters
    ax.set_title(plottitle,fontsize=10)
    ax.tick_params(axis='x',labelsize=10,labelbottom=False)
    ax.tick_params(axis='y',labelsize=10,labelleft=False)

    #- Add optional arguments
    if heatmap: cmap = heatmap
    else: cmap = 'OrRd'

    #- Generate patch plot
    patch=ax.pcolor(vals.reshape(grid[0],grid[1]),cmap=cmap)

    return patch

def ql_2dplot(ax,xvals,yvals,plottitle,xtitle,ytitle,xlim=None,ylim=None):
    """
    Make 2d plot of specific metrics provided in configuration file

    Args:
        ax: matplotlib subplot
        xvals: QA metric to be plotted along the xaxis
        yvals: QA metric to be plotted along the yaxis
        plottitle: plot title from configuration file
        xtitle: x axis label
        ytitle: y axis label
    Optional:
        xlim: list containing x range (i.e. [x_lo,x_hi])
        ylim: list containing y range (i.e. [y_lo,y_hi])

    Returns:
        matplotlib sublot containing plotted metrics
    """
    #- Set title and axis labels
    ax.set_title(plottitle,fontsize=10)
    ax.set_xlabel(xtitle,fontsize=10)
    ax.set_ylabel(ytitle,fontsize=10)

    #- Add optional arguments
    if xlim: ax.set_xlim(xlim[0],xlim[1])
    if ylim: ax.set_ylim(ylim[0],ylim[1])

    #- Generate 2d plot
    ax.plot(xvals,yvals)

    return ax

def ql_3dplot(ax,xvals,yvals,zvals,plottitle,xtitle,ytitle,zlim=None,heatmap=None):
    """
    Make 3d scatter plot of specific metrics provided in configuration file

    Args:
        ax: matplotlib subplot
        xvals: QA metric to be plotted along the xaxis
        yvals: QA metric to be plotted along the yaxis
        zvals: QA metric to be plotted
        plottitle: plot title from configuration file
        xtitle: x axis label
        ytitle: y axis label
    Optional:
        zlim: list containing scatter plot range (i.e. [z_lo,z_hi])

    Returns:
        matplotlib sublot containing plotted metrics
    """
    #- Setup title and axies labels
    ax.set_title(plottitle,fontsize=10)
    ax.set_xlabel(xtitle,fontsize=10)
    ax.set_ylabel(ytitle,fontsize=10)

    #- Add optinoal arguments
    if heatmap: cmap = heatmap
    else: heatmap = 'bwr'
    if zlim: vmin,vmax = zlim[0],zlim[1]
    else: vmin,vmax = np.min(zvals), np.max(zvals)

    #- Generate 3d scatter plot
    scatter=ax.scatter(xvals,yvals,c=zvals,cmap=heatmap,vmin=vmin,vmax=vmax)

    return scatter
