import pyspark.sql.functions as F
from pyspark.sql.types import *

import json
import datetime
import pytz
import six
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go
import pickle


class Utilerias(object):
    def __init__(self):
        pass
    
        # Función que realiza la lectura del JSON
    #-----------------------------------------------------------------------------
    def fnCargaJson(nexus_url):
        with open(nexus_url) as json_file: properties = json.load(json_file)
        return properties
    
    # Función para devolver la hora actual
    #-----------------------------------------------------------------------------
    def fnHoraActual():
        sHoraActual= datetime.datetime.now(pytz.timezone('Mexico/General')).strftime("%H:%M:%S %p")
        return sHoraActual
    
    # Función para devolver tiempo transcurrido
    #-----------------------------------------------------------------------------
    def fnTiempoTranscurrido (HoraInicio, HoraFin):
        sTiempoTranscurrido = dt.strptime(HoraFin[:8], '%H:%M:%S') - dt.strptime(HoraInicio[:8], '%H:%M:%S')
        return str(sTiempoTranscurrido)
    
    # Función para convertir una fecha String a Date
    #-----------------------------------------------------------------------------
    def fnConvierteFecha (fechaConvertir):
        if (fechaConvertir is None):
            return None
        else:
            sTipoFecha  = fechaConvertir[2:3]
            if (sTipoFecha == "/"):
                return dt.strptime(fechaConvertir, '%d/%m/%Y')
            else:
                sTipoFecha  = fechaConvertir[4:5]
            if (sTipoFecha == "-"):
                return dt.strptime(fechaConvertir, '%Y-%m-%d')
            else:
                "9999-01-01"


    # Function to create images from Pandas Data Frames
    #-----------------------------------------------------------------------------
    def render_mpl_table(data, header_columns=0, col_width=3.0, row_height=0.5, font_size=12, **kwargs):
        """
        Function to render a data frane to image.
        """
        # Parametros que dan diseño a la tabla
        header_color='#40466e'
        row_colors=['#f1f1f2','w']
        edge_color='w'
        bbox=[0, 0, 1, 1]
        ax=None
        
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')
        
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)
        
        for k, cell in  six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
        return ax

    
    def save_tableimage(table, path_file, header, width):
        """
        Function to save image into a path defined by users.
        """
        table_fig = Utilerias.render_mpl_table(table, header_columns=header, col_width=width)
        table_fig = table_fig.get_figure()
        table_fig.savefig(path_file, dpi=200)
        
    # Information value graph
    #-----------------------------------------------------------------------------
    def graph_iv(inf_values):
        """
        Check point number 2. Interactive graph the information value on each
        percentil for all variables of interest.
        
        :param str inf_values: List of objects from infval_all() function
        :return: Interactive graph with plotly
        """
        cant_vars = list(inf_values[0].columns[1:])
        data = []
        for i in cant_vars:
            # Create traces
            trace = go.Scatter(
                x = list(inf_values[0]["percentil"]),
                y = list(inf_values[0][i]),
                mode = 'lines+markers',
                name = i)
            data.append(trace)
        
        # Edit the layout
        layout = dict(title = 'Segmentation Analysis - Information Value',
                      xaxis = dict(title = 'Percentils'),
                      yaxis = dict(title = 'Information Value'),
                      )

        # Plot and embed in ipython notebook!
        fig = dict(data=data, layout=layout)

        # Graph
        return iplot(fig)
    
    # Best scores for each generation graph (Genetic algorithm)
    #-----------------------------------------------------------------------------
    def plot_scores(GeneticAlgorithm):
        plt.figure(figsize=(8,5))
        plt.plot(GeneticAlgorithm[1], label='Best', marker="o")
        plt.plot(GeneticAlgorithm[2], label='Average', marker="o")
        plt.legend()
        plt.ylabel('Scores')
        plt.xlabel('Generation')
        plt.title("Genetic algorithm evolution", fontsize=16)
        plt.show()
    
    # Split dataframe into train and test samples
    #-----------------------------------------------------------------------------
    def balanc_split(table, var_obj, train_size, seed):
        """
        Returns a stratified sample without replacement based on the fraction given on each stratum.
        """
        train = table.sampleBy(var_obj, fractions={0: train_size, 1: train_size}, seed=seed)
        test = table.subtract(train)
        train = train.select(*train.columns,(F.lit("train")).alias("sample"))
        test = test.select(*test.columns,(F.lit("test")).alias("sample"))
        table_all = train.union(test)

        return table_all
    
    #-----------------------------------------------------------------------------
    def sample_function(table, type_sample, var_obj, withReplacement, proportion, set_seed):
        """
        Returns a sampled subset of this DataFrame.
        """
        if type_sample == "simple":
            sample = table.sample(False, fraction=proportion, seed=0)
        elif type_sample == "stratified":
            sample = table.sampleBy(var_obj, {0: proportion, 1: proportion}, set_seed)
        else:
            sample = print("Choose type of sampling.")
        
        return sample
    
    #-----------------------------------------------------------------------------
    def mean_analysis_plot(mean_analysis, legend, path):
        """
        Return plots for mean analysis of the variables
        """
        list_variables = mean_analysis["variable"].unique()
        var_cohorte = mean_analysis.columns[0]
        for k in list_variables:
            # Tools for plot
            table_mean = mean_analysis[mean_analysis["variable"] == k].reset_index()
            aux_mean = table_mean["mean"].mean()
            aux_std = table_mean["mean"].std()
            lim_min = aux_mean - (5*aux_std)
            lim_max = aux_mean + (5*aux_std)
            largo_cohortes = len(table_mean[var_cohorte])
            # Plot
            ax = table_mean.plot(kind="line", y="mean", marker="o", rot=90, grid=False, figsize=(10,5),
                                 title=str("Temporal analysis (mean) of " + k), color="tab:blue")
            ax.set_xlabel("Cohortes", size=12)
            ax.set_ylabel("Mean", size=12)
            plt.xticks(range(largo_cohortes), table_mean[var_cohorte], size='small')
            ax.set_ylim([lim_min, lim_max])
            # Labels for each point into line graph
            if legend == True:
                for i,j in table_mean["mean"].items():
                    ax.annotate(str(round(j,3)), xy=(i, j+0.01))
            # Saving plot
            fig = ax.get_figure()
            fig.savefig(path + "mean_analysis_" + str(k) + ".png", dpi=200)

    #-----------------------------------------------------------------------------
    def median_analysis_plot(median_analysis, legend, path):
        """
        Return plots for median analysis of the variables
        """
        list_variables = median_analysis["variable"].unique()
        var_cohorte = median_analysis.columns[0]
        
        for k in list_variables:
            # Tools for plot
            table_median = median_analysis[median_analysis["variable"] == k].reset_index()
            largo_cohortes = len(table_median[var_cohorte])
            ax = table_median.plot(kind="line", y="q_25", color="tab:red", rot=90, grid=False, figsize=(10,5),
                                   linestyle="--", title=str("Temporal analysis (quantiles) of "+k),
                                   label="Quantil 0.25")
            table_median.plot(y="median", kind="line", ax=ax, color="tab:blue", marker="o", rot=90, grid=False)
            table_median.plot(y="q_75", kind="line", ax=ax, color="tab:orange", rot=90, grid=False,
                              linestyle="--", label="Quantil 0.75")
            ax.set_xlabel("Cohortes", size=12)
            ax.set_ylabel("Median", size=12)
            plt.xticks(range(largo_cohortes), table_median[var_cohorte], size='small')
            # Labels for each point
            if legend == True:
                for i,j in table_median["median"].items():
                    ax.annotate(str(round(j,3)), xy=(i, j+0.01))
            # Saving plot
            fig = ax.get_figure()
            fig.savefig(path + "median_analysis_" + str(k) + ".png", dpi=200)
    
    #-----------------------------------------------------------------------------
    def categ_analysis_plot(categ_analysis, path):
        """
        Return plots of categoric analysis for each categoric variable included
        """
        list_variables = categ_analysis[0]["variable"].unique()
        k = 0
        for i in list_variables:
            pivot_categ = categ_analysis[1][k]
            k = k + 1
            num = 0
            plt.figure(figsize=(15,5))
            lColumns = len(pivot_categ.columns)
            for j in pivot_categ:
                num = num + 1
                plt.subplot(1, lColumns, num)
                aux_mean = pivot_categ[j].mean()
                aux_std = pivot_categ[j].std()
                lim_min = aux_mean - (10*aux_std)
                lim_max = aux_mean + (5*aux_std)
                ax = plt.plot(pivot_categ[j], marker='o', color="tab:blue")
                plt.title("Category: "+str(j), loc='center', fontsize=12, fontweight=0, color="tab:blue")
                plt.ylim([lim_min,lim_max])
                plt.suptitle("Frequency evolution of each category ::: Variable: "+str(i), fontsize=13,
                             fontweight="bold", color="black", style="normal", y=0.98) # normal, italic or oblique
                plt.xlabel("Cohortes");
            # Saving plot
            plt.savefig(path + "categ_analysis_" + str(i) + ".png", dpi=200)


    # Plot fucntion for tm_quantil analysis
    #-----------------------------------------------------------------------------
    def tm_quantil_plot(tm_quantil, legend, path):
        """
        Return plots of bad rates by quentiles for each numeric variable included
        """
        list_variables = tm_quantil[0]["variable"].unique()
        k = 0
        for i in list_variables:
            tm_plot = tm_quantil[0][tm_quantil[0]["variable"]==i].reset_index()
            # List for axis X
            listValuesQ = tm_quantil[1][k]
            k = k + 1
            list_ejex = list(listValuesQ)
            list_ejex = [round(x, 2) for x in list_ejex]
            list_ejex.pop(0)
            # Plot
            ax = tm_plot.plot(kind="line", x="percentil", y="tm", marker='o', figsize=(10,5), rot=90)
            plt.title("Evolution of Default rate for the numeric variable '" + i + "'",
                      loc='center', fontsize=12, fontweight="bold", color="black")
            plt.xticks(range(len(list_ejex)), list_ejex, size='small')
            ax.set_ylabel("Default rate", size=12)
            ax.set_xlabel("Percentil", size=12)
            # Labels for each point into line graph
            if legend == True:
                for m,j in tm_plot["tm"].items():
                    ax.annotate(str(round(j,3)), xy=(m, j+0.01))
            
            # Saving plot
            fig = ax.get_figure()
            fig.savefig(path + "tm_quantil_" + str(i) + ".png", dpi=200)


    # Plot fucntion for tm_quantil_categ analysis
    #-----------------------------------------------------------------------------
    def tm_quantil_categ_plot(tm_quantil_categ, legend, path):
        """
        Return plots of bad rates by quentiles for each categoric variable included
        """
        list_variables = tm_quantil_categ["variable"].unique()
        for i in list_variables:
            # Plot
            ctm_plot = tm_quantil_categ[tm_quantil_categ["variable"]==i].reset_index()
            ax = ctm_plot.plot(kind="bar", x="category", y="total", figsize=(10,5), rot=0, color="lightgray", legend=False)
            plt.title("Default rate by category for the variable '" + i + "'",
                      loc='center', fontsize=12, fontweight="bold", color="black")
            #plt.xticks(range(len(list_ejex)), list_ejex, size='small')
            ax.set_ylabel("Frequency", size=12)
            ax.set_xlabel("Categories", size=12)
            aux = ctm_plot["tm"].plot(kind = "line", rot=0, marker='o', secondary_y=True, ax=ax, figsize=(10,5), color = "steelblue")
            aux.set_ylabel("Default rate", size=12)
            ylim_mean = ctm_plot["tm"].mean()
            ylim_std = ctm_plot["tm"].std()
            ylim_min = ylim_mean - (5*ylim_std)
            ylim_max = ylim_mean + (5*ylim_std)
            aux.set_ylim([ylim_min,ylim_max])
            # Labels for each point into line graph
            if legend == True:
                for m,j in ctm_plot["tm"].items():
                    aux.annotate(str(round(j,3)), xy=(m, j))
            # Saving plot
            fig = ax.get_figure()
            fig.savefig(path + "tm_quantil_categ_" + str(i) + ".png", dpi=200)
    
    #-----------------------------------------------------------------------------
    def psi_plot(psi_table, path_file):
        """
        Plot output of the function 'psi_table'.
        
        :param list psi_table: list of outputs of the function 'psi_table'
        :return: list with [table_psi]
        
        Example:
            >>> b = psi_table(spark_table, var_interes, "cohorte", 201501, 5)
            >>> c = psi_graph(b)
        """
        cohorte_name = psi_table[0]
        colors = plt.cm.PuBu(np.linspace(0, 1, len(psi_table[1])+1))
        table_psi = psi_table[3]#.toPandas()
        Lst = []
        for i in psi_table[4]:
            table_filter = table_psi[table_psi["Variable"] == i]
            table_filter = table_filter.pivot(index=cohorte_name, columns="Q", values="Proportion")
            aux = table_filter.plot(kind="bar", stacked=True, legend=False, color=colors, figsize=(9,5), grid=False,
                                    title=str("PSI of "+i));
            # Save plot
            fig = aux.get_figure()
            fig.savefig(path_file + "psi_" + str(i) + ".png", dpi=200)
            Lst.append(aux)

        return Lst
    
    #-----------------------------------------------------------------------------
    def def_universe_plot(def_universe_outputs, path_file):
        """
        Plotting functions of Puntos de Control 2.
        """
        # Outputs of "def_universe"
        dimension = def_universe_outputs[0]
        rango_cohortes = def_universe_outputs[1]
        periodo = def_universe_outputs[2]
        tabla_tm = def_universe_outputs[3]
        
        # Save table of "dimension"
        tabla_fig = Utilerias.render_mpl_table(dimension, header_columns=0, col_width=3)
        tabla_fig = tabla_fig.get_figure()
        tabla_fig.savefig(path_file + "dimension.png", dpi=200)
        
        # Save table of "rango_cohorte"
        tabla_fig = Utilerias.render_mpl_table(rango_cohortes, header_columns=0, col_width=3)
        tabla_fig = tabla_fig.get_figure()
        tabla_fig.savefig(path_file + "rango_cohortes.png", dpi=200)
        
        # Save table of "periodo"
        tabla_fig = Utilerias.render_mpl_table(periodo, header_columns=0, col_width=3)
        tabla_fig = tabla_fig.get_figure()
        tabla_fig.savefig(path_file + "periodo.png", dpi=200)
    
        # Save plot of "periodo"
        periodo_plot = periodo.plot(kind="bar", x="Year", y="count", color="tab:blue", rot=0,
                                    grid=False, title="Count of registers per year", figsize=(8,5))
        periodo_plot.set_xlabel("Year", fontsize=12)
        periodo_plot.set_ylabel("Count", fontsize=12)
        
        totals = []
        for i in periodo_plot.patches:
            totals.append(i.get_height())
        total = sum(totals)
        for i in periodo_plot.patches:
            periodo_plot.text(i.get_x()+0.05, i.get_height()+10, # the "-200" is the position of labels
                              str(round((i.get_height()/total) * 100,2))+"%", fontsize=12, color="black")
        fig = periodo_plot.get_figure()
        fig.savefig(path_file + "periodo_plot.png", dpi=200)
        
        # Save table of "tabla_tm"
        tabla_fig = Utilerias.render_mpl_table(tabla_tm, header_columns=0, col_width=3)
        tabla_fig = tabla_fig.get_figure()
        tabla_fig.savefig(path_file + "tabla_tm.png", dpi=200)

    #-----------------------------------------------------------------------------
    def init_mature_plot(init_mature_list, path_file):
        """
        This function ploting the tables calculated in the function 'init_mature'.
        
        :param str init_mature_list: Output of function 'init_mature'
        :return: Graphs and list with [distseg_plot, distsample_plot, distsegsam_plot, distrib_client_plot, distrib_samcli_plot]
        
        Example:
            >>> w = init_mature(dummyBal, "segmento", "sample", "incmpl_d")
            >>> w1 = init_mature_graph(w)
        """
        #---------------------------------------------
        # Segment distribution GRAPH
        distrib_seg = init_mature_list[0]#.toPandas()
        distseg_plot = distrib_seg.plot(kind="bar", x="segment", y="count", color="lightsteelblue", rot=0, grid=False,
                                        title="Segments distribution", figsize=(8,5))
        distseg_plot.set_xlabel("Segments", fontsize=12)
        distseg_plot.set_ylabel("Count", fontsize=12)
        # create a list to collect the plt.patches data
        totals = []
        # find the values and append to list
        for i in distseg_plot.patches:
            totals.append(i.get_height())
        # set individual bar lables using above list
        total = sum(totals)
        # set individual bar lables using above list
        for i in distseg_plot.patches:
            # get_x pulls left or right; get_height pushes up or down
            distseg_plot.text(i.get_x()+0.10, i.get_height()+0.10,
                              str(round((i.get_height()/total) * 100,2))+"%", fontsize=12, color="black")
        fig = distseg_plot.get_figure()
        fig.savefig(path_file + "distseg_plot.png", dpi=200)

        #---------------------------------------------
        # Sample distribution GRAPH
        distrib_sample = init_mature_list[1]#.toPandas()
        distsample_plot = distrib_sample.plot(kind="bar", x="sample", y="count", color="steelblue", rot=0, grid=False,
                                              title="Sample distribution", figsize=(8,5))
        distsample_plot.set_xlabel("Sample", fontsize=12)
        distsample_plot.set_ylabel("Count", fontsize=12)

        totals = []
        for i in distsample_plot.patches:
            totals.append(i.get_height())
        total = sum(totals)
        for i in distsample_plot.patches:
            distsample_plot.text(i.get_x()+0.10, i.get_height()+0.10,
                                 str(round((i.get_height()/total) * 100,2))+"%", fontsize=12, color="black")
        fig = distsample_plot.get_figure()
        fig.savefig(path_file + "distsample_plot.png", dpi=200)

        #---------------------------------------------
        # Segment and sample distribution GRAPH
        distrib_segsam = init_mature_list[2]#.toPandas()
        distrib_segsam["segm_sample"] = distrib_segsam["segment"].map(str) + "_" + distrib_segsam["sample"].map(str)
        distsegsam_plot = distrib_segsam.plot(kind="bar", x="segm_sample", y="count", color="tab:blue", rot=0,
                                              grid=False, title="Segment and sample distribution", figsize=(8,5))
        distsegsam_plot.set_xlabel("Segment and sample", fontsize=12)
        distsegsam_plot.set_ylabel("Count", fontsize=12)

        totals = []
        for i in distsegsam_plot.patches:
            totals.append(i.get_height())
        total = sum(totals)
        for i in distsegsam_plot.patches:
            distsegsam_plot.text(i.get_x()+0.10, i.get_height()+0.10,
                                 str(round((i.get_height()/total) * 100,2))+"%", fontsize=12, color="black")
        fig = distsegsam_plot.get_figure()
        fig.savefig(path_file + "distsegsam_plot.png", dpi=200)


        #---------------------------------------------
        # Distribution by client GRAPH
        distrib_client = init_mature_list[3]#.toPandas()
        distrib_client_plot = distrib_client.plot(kind="bar", x="incmpl", y="count", color="gray", rot=0,
                                                      grid=False, title="Client distribution", figsize=(8,5))
        distrib_client_plot.set_xlabel("Client", fontsize=12)
        distrib_client_plot.set_ylabel("Count", fontsize=12)

        totals = []
        for i in distrib_client_plot.patches:
            totals.append(i.get_height())
        total = sum(totals)
        for i in distrib_client_plot.patches:
            distrib_client_plot.text(i.get_x()+0.10, i.get_height()+0.10,
                                     str(round((i.get_height()/total) * 100,2))+"%", fontsize=12, color="black")
        fig = distrib_client_plot.get_figure()
        fig.savefig(path_file + "distrib_client_plot.png", dpi=200)

        #---------------------------------------------
        # Distribution by sample and client GRAPH
        distrib_samcli = init_mature_list[4]#.toPandas()
        distrib_samcli["sample_cli"] = distrib_samcli["sample"].map(str) + "_" + distrib_samcli["incmpl"].map(str)
        distrib_samcli_plot = distrib_samcli.plot(kind="bar", x="sample_cli", y="count", color="gold", rot=0,
                                                  grid=False, title="Sample and client distribution", figsize=(8,5))
        distrib_samcli_plot.set_xlabel("Sample and client", fontsize=12)
        distrib_samcli_plot.set_ylabel("Count", fontsize=12)

        totals = []
        for i in distrib_samcli_plot.patches:
            totals.append(i.get_height())
        total = sum(totals)
        for i in distrib_samcli_plot.patches:
            distrib_samcli_plot.text(i.get_x()+0.10, i.get_height()+0.10,
                                     str(round((i.get_height()/total) * 100,2))+"%", fontsize=12, color="black")
        fig = distrib_samcli_plot.get_figure()
        fig.savefig(path_file + "distrib_samcli_plot.png", dpi=200)


        Lst = [distseg_plot, distsample_plot, distsegsam_plot, distrib_client_plot, distrib_samcli_plot]

        return Lst

    #-----------------------------------------------------------------------------
    def evol_cohorte_plot(evol_cohorte_list, path_file):
        """
        Plotting of evolution of default ratios through the differents cohortes on data base.
        """
        # Plot
        Lst = []
        for i in range(len(evol_cohorte_list)):
            if len(evol_cohorte_list) > 1:
                evol_cohorte = evol_cohorte_list[i]
                ax = evol_cohorte.plot(kind="bar", x="cohorte", y="total", color="lightgray", rot=90, alpha=0.5,
                                       grid=False, title="Evolution of TM for each cohorte in segment "+str(i+1), figsize=(8,5))
                ax.set_xlabel("Cohorte", fontsize=12)
                ax.set_ylabel("Count (bar)", fontsize=12)
                aux = evol_cohorte["tm"].plot(kind="line", rot=90, marker='o', secondary_y=True, ax=ax,
                                              figsize=(10,5), color="steelblue")
                aux.set_ylabel("TM (line)")
            
                # Save plot
                fig = aux.get_figure()
                fig.savefig(path_file + "evol_tm_cohorte_seg" + str(i+1) + ".png")
                Lst.append(aux)
            else:
                evol_cohorte = evol_cohorte_list[i]
                ax = evol_cohorte.plot(kind="bar", x="cohorte", y="total", color="lightgray", rot=90, alpha=0.5,
                                       grid=False, title="Evolution of TM for each cohorte", figsize=(8,5))
                ax.set_xlabel("Cohorte", fontsize=12)
                ax.set_ylabel("Count (bar)", fontsize=12)
                aux = evol_cohorte["tm"].plot(kind="line", rot=90, marker='o', secondary_y=True, ax=ax,
                                              figsize=(10,5), color="steelblue")
                aux.set_ylabel("TM (line)")

                # Save plot
                fig = aux.get_figure()
                fig.savefig(path_file + "evol_tm_cohorte.png")
                Lst.append(aux)
        
        return Lst

    #-----------------------------------------------------------------------------
    def tmora_seg_table(tmora_seg, path_file):
        """
        Saving tmora_seg table.
        """
        seg_tmora_pandas = tmora_seg
        # Saving table "seg_tmora"
        tabla_fig = Utilerias.render_mpl_table(seg_tmora_pandas, header_columns=0, col_width=2)
        tabla_fig = tabla_fig.get_figure()
        tabla_fig.savefig(path_file + "tmora_seg.png", dpi=200)

    #-----------------------------------------------------------------------------
    def save_load_objects(pickle_object, analysis_object, action):
        if action == "save":
            with open(pickle_object, 'wb') as f:
                pickle.dump(analysis_object, f)
        elif action == "load":
            with open(pickle_object, "rb") as f:
                load_info = pickle.load(f)
            return load_info
        else:
            print("Define an action (save or load).")

