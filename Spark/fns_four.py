import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from Utilerias import *
from pyspark.ml.feature import Bucketizer
from pyspark.mllib.stat import Statistics #Correlation anaysis
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.storagelevel import StorageLevel

import numpy as np
import pandas as pd
import sys #Progress bar

class cp_four(object):
    """
    - Depuración de los datos
        - Analisis de correlación
        - Variabilidad
        - Analisis de valores nulos
        - Deteccion de outliers
    - Creacion de nuevas variables
        - Ratios sectorizados
    """
    def __init__(self, sqlContext):
        self.sqlContext = sqlContext
        pass
    
    def correlation_analysis(self, table, params):
        """
        Create a correlation matrix with ginis from numeric columns.
        """
        # Drop columns in the black list
        lColumns = [x for x in params["var_interes"] if x not in params["black_list"]]
        
        str_cols = [item[0] for item in table.dtypes if item[1].startswith('string')]
        num_cols = [x for x in lColumns if x not in str_cols]
        table_filter = table.select(num_cols)#.na.drop(how = "any")
        features = table_filter.rdd.map(lambda row: row[0:])
        corr_mat = Statistics.corr(features, method="pearson")
        corr_df = pd.DataFrame(corr_mat)
        corr_df.index, corr_df.columns = table_filter.columns, table_filter.columns
        corr_df = corr_df.where(np.triu(np.ones(corr_df.shape)).astype(np.bool))
        table_cor = corr_df.stack().reset_index()
        table_cor.columns = ["row","column","correlation"]
        table_cor["correlation"] = abs(table_cor["correlation"])
        table_cor = table_cor.sort_values(by=["correlation"], ascending=False)
        table_cor = table_cor[table_cor["row"] != table_cor["column"]]
        table_cor = table_cor.loc[table_cor["correlation"] > params["lim_corr"]]
        list_unique = list(table_cor["row"].unique()) + list(table_cor["column"].unique())
        list_unique = list(set(list_unique))
        if len(list_unique) > 0:
            # Empty data frame of "gini_all" to storage the ginis calculated
            gini_all = pd.DataFrame()
            # Loop to calculate the gini by variable
            k = 0
            long_k = len(list_unique)
            for i in list_unique:
                k = k+1
                lcolumns = [i,params["var_obj"]]
                # Model
                vectorAssembler = VectorAssembler(inputCols = [i], outputCol = "features")
                v_table = vectorAssembler.transform(table_filter.select(lcolumns))
                v_table = v_table.select(["features", params["var_obj"]])
                v_table = v_table.select(*v_table.columns, F.col(params["var_obj"]).alias("label"))
                # Logistic model
                lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
                lrModel = lr.fit(v_table)
                # areaUnderROC
                auc_iter = lrModel.summary.areaUnderROC
                if auc_iter < 0.5:
                    gini_iter = pd.DataFrame(data=[(i, 2*(1-auc_iter)-1, "u")], columns=("variable","gini","tndnc"))
                else:
                    gini_iter = pd.DataFrame(data=[(i, 2*auc_iter-1, "d")], columns=("variable","gini","tndnc"))

                # Append to data frame
                gini_all = pd.concat([gini_all, gini_iter], axis=0, ignore_index=True)
                # Progress bar
                sys.stdout.write('\r')
                sys.stdout.write("Iteration of correlation (gini): {0}/{1}\tProgress: {2:.2f}%".format(k,
                                                                                                       long_k,
                                                                                                       (100/long_k)*k))
                sys.stdout.flush()

            # Create a data frame with correlations and ginis
            table_cor = pd.merge(table_cor, gini_all[["variable","gini"]], left_on="row", right_on="variable", how="left") \
                .drop("variable", axis=1).rename(columns={"gini": "row_gini"})
            table_cor = pd.merge(table_cor, gini_all[["variable","gini"]], left_on="column", right_on="variable", how="left") \
                .drop("variable", axis=1).rename(columns={"gini": "column_gini"})
            table_cor["drop"] = np.where(table_cor["row_gini"] >= table_cor["column_gini"], table_cor["column"], table_cor["row"])
            corr_list = table_cor["drop"].tolist()
            corr_list = list(set(corr_list))
        
        else:
            print("No correlations upper than the limit",params["lim_corr"])
            gini_all = pd.DataFrame()
            corr_list = []
        
        return [table_cor, gini_all, corr_list]


    def variability_analysis(self, table, params):
        """
        Analysis of variability with IQR.
        """
        # Drop columns in the black list and variable objective
        drop_init = params["black_list"] + [params["var_obj"]]
        lColumns = [x for x in params["var_interes"] if x not in drop_init]
        
        table_filter = table.select(*lColumns)
        # Type of variables
        str_cols = [item[0] for item in table_filter.dtypes if item[1].startswith('string')]
        num_cols = [x for x in lColumns if x not in str_cols]
        # Loop to calculate Q1 and Q3
        print("\n-----")
        k = 0
        long_k = len(num_cols)
        summary_all = pd.DataFrame()
        for i in num_cols:
            k=k+1
            # Filter special values
            try:
                table_iter = table_filter.select(i).filter(~F.col(i).isin(params["special_values"][i]))
            except:
                table_iter = table_filter.select(i)
            # Quantiles calculation
            quantiles = F.expr("percentile_approx(" + i + ", array(0.00, 0.25, 0.5, 0.75, 1.00))")
            summary_iter = table_iter.agg(quantiles.alias("Quantiles"))
            summary_iter = summary_iter.select(summary_iter.Quantiles[0].alias("q_0"),
                                               summary_iter.Quantiles[1].alias("q_25"),
                                               summary_iter.Quantiles[2].alias("median"),
                                               summary_iter.Quantiles[3].alias("q_75"),
                                               summary_iter.Quantiles[4].alias("q_100")).toPandas()
            summary_iter["variable"] = i
            summary_all = pd.concat([summary_all, summary_iter], axis=0, ignore_index=True)
            # Progress bar
            sys.stdout.write('\r')
            sys.stdout.write("Iteration of variability and none values: {0}/{1}\tProgress: {2:.2f}%".format(k,
                                                                                                            long_k,
                                                                                                            (100/long_k)*k))
            sys.stdout.flush()
        
        summary_all["IQR"] = summary_all["q_75"] - summary_all["q_25"]
        summary_all["variability"] = np.where(summary_all["q_75"] - summary_all["q_25"] == 0, 0,
                                              summary_all["IQR"] / (summary_all["q_100"] - summary_all["q_0"]))*100
        summary_all = summary_all.sort_values("variability", ascending=False)
        summary_filter = summary_all[summary_all["variability"] < params["lim_variab"]]
        summary_filter = summary_filter[["variable","q_0","q_25","median","q_75","q_100","IQR","variability"]]
        variability_list = list(summary_filter["variable"])
        
        return [summary_all, variability_list]


    def ExtractFeatureImp(self, featureImp, dataset, featuresCol):
        """
        Extract feature importance for each variable with random forest algorithm.
        """
        list_extract = []
        for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
            list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
        varlist = pd.DataFrame(list_extract)
        varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
        varlist = varlist.sort_values('score', ascending = False)
        varlist = varlist.reset_index(drop=True)
        return varlist


    def analysis_rf(self, table, params):
        """
        Estimate of the importance of each feature.
        
        Each feature's importance is the average of its importance across all trees in the ensemble
        The importance vector is normalized to sum to 1. This method is suggested by Hastie et al.
        (Hastie, Tibshirani, Friedman. "The Elements of Statistical Learning, 2nd Edition." 2001.)
        and follows the implementation from scikit-learn.
        """
        lColumns = [x for x in params["var_interes"] if x not in params["black_list"]]
        str_cols = [item[0] for item in table.dtypes if item[1].startswith('string')]
        num_cols = [x for x in lColumns if x not in str_cols]
        # Filter table and list
        table_filter = table.select(num_cols)#.na.drop(how = "any")
        num_cols.remove(params["var_obj"])
        # Random Forest algorithm
        vectorAssembler = VectorAssembler(inputCols = num_cols, outputCol = "features")
        v_table = vectorAssembler.transform(table_filter.select(num_cols+[params["var_obj"]]))
        v_table = v_table.select(["features", params["var_obj"]])
        v_table = v_table.select(*v_table.columns, F.col(params["var_obj"]).alias("label"))
        print("\n-----")
        # Build the model
        rf = RandomForestClassifier(numTrees = params["analysis_rf"]["num_trees"],
                                    maxDepth = params["analysis_rf"]["max_depth"],
                                    labelCol = "label",
                                    seed = params["analysis_rf"]["rf_seed"])
        model_rf = rf.fit(v_table)
        feature_importance = model_rf.featureImportances
        feature_importance_DF = cp_four.ExtractFeatureImp(self, feature_importance, v_table, "features")
        aux_fi = feature_importance_DF["name"].tail(params["analysis_rf"]["lim_imprf"])
        list_fi = list(aux_fi)
        print("Feature importance by RF is done.")
        
        return [feature_importance, feature_importance_DF, list_fi]


    def matrix_fs(self, params, a, b, c):
        """
        Matrix of results for the feature selection analysis.
        """
        lcolumns = [x for x in params["var_interes"] if x not in [params["var_obj"]]+params["black_list"]]
        matrix_all = pd.DataFrame(lcolumns)
        matrix_all.columns = ["variable"]
        
        df_aux = pd.DataFrame(a[2])
        df_aux.columns = ["variable"]
        df_aux["correlation"] = 1
        matrix_all = pd.merge(matrix_all, df_aux, on="variable", how="left")
        
        df_aux = pd.DataFrame(b[1])
        df_aux.columns = ["variable"]
        df_aux["variability"] = 1
        matrix_all = pd.merge(matrix_all, df_aux, on="variable", how="left")
        
        df_aux = pd.DataFrame(c[2])
        df_aux.columns = ["variable"]
        df_aux["low_importance"] = 1
        matrix_all = pd.merge(matrix_all, df_aux, on="variable", how="left")
        matrix_all = matrix_all.fillna(0)
        matrix_all["count"] = matrix_all.sum(axis=1)
        # Choose rows with at least one number "1.0"
        matrix_all_filter = matrix_all[(matrix_all.iloc[:,:] == 1.0).any(axis=1)]
        matrix_all_filter = matrix_all_filter.sort_values("count", ascending=False).reset_index(drop=True)
        
        # Add "correlation with" column to "matrix_all_filter"
        aux_corr = a[0]
        aux_corr_a = aux_corr.groupby(["row"]).column.apply(" & ".join).reset_index()
        aux_corr_b = aux_corr.groupby(["column"]).row.apply(" & ".join).reset_index()
        matrix_all_filter = pd.merge(matrix_all_filter, aux_corr_a[["row","column"]], left_on="variable",
                                     right_on="row", how="left")
        matrix_all_filter = matrix_all_filter.drop(["row"], axis=1).rename(columns={"column": "aux1"})
        matrix_all_filter = pd.merge(matrix_all_filter, aux_corr_b[["column","row"]], left_on="variable",
                                     right_on="column", how="left")
        matrix_all_filter = matrix_all_filter.drop(["column"], axis=1).rename(columns={"row": "aux2"})
        matrix_all_filter["correlation with"] = np.where(matrix_all_filter["aux1"].isnull(), matrix_all_filter["aux2"],
                                                         np.where(matrix_all_filter["aux2"].isnull(), matrix_all_filter["aux1"],
                                                                  matrix_all_filter["aux1"] + " & " +  matrix_all_filter["aux2"]))
        matrix_all_filter = matrix_all_filter.drop(["aux1","aux2"], axis=1)
        
        return [matrix_all, matrix_all_filter]


    def model_selection(self, table, params, deep):
        """
        Function for feature selection and modeling.
        """
        if deep == "selection":
            lcolumns = [x for x in table.columns if x not in params["black_list"]]
            table_fs = table.select(*lcolumns)
            table_fs.persist(StorageLevel.MEMORY_AND_DISK)
            
            a = cp_four.correlation_analysis(self, table_fs, params)
            b = cp_four.variability_analysis(self, table_fs, params)
            c = cp_four.analysis_rf(self, table_fs, params)
            d = cp_four.matrix_fs(self, lcolumns, a, b, c)
            return [a, b, c, d]
            table_fs.unpersist()
        
        elif deep == "modeling":
            a = 1
            b = 1
            c = 1
            return [a, b, c]
        
        elif deep == "all":
            print("Working on it...")
            return None
        
        else:
            print("This depth does not exist.")
            return None



class cp_two(object):
    """
    - Segmentation analysis
    - Initial distribution and mature
        - Sample distribution
        - Mature filters
    - Sample selection of modeling
    """
    def __init__(self, sqlContext):
        self.sqlContext = sqlContext
        pass
    
    def def_universe(self, table, params):
        """
        Functions of Puntos de Control 2, this functions calculate: 1) dimention table;
        2) cohortes table, 3) analysis of period (year), and 4) graphs of each analysis.
        
        :param str table: Database to analyze
        :param str cohorte: cohorte column in database
        :param str var_obj: var_obj in database, also called incmpl or bad
        :return: list with [dimension, rango_cohortes, periodo, tabla_tm]
        
        Example:
            >>> y = mature_analysis.def_universe(spark_table, "cohorte", "incmpl_d")
        """
        # Variables from dictionary
        var_cohorte = params["checkpoints2"]["var_cohorte"]
        var_obj = params["var_obj"]
        
        # Dimention table
        count_df = table.count()
        columns_df = len(table.columns)
        dimension = pd.DataFrame(data=[(count_df,columns_df)], columns=("Number of registers","Number of columns"))
        
        # Cohortes table
        rango_cohortes = table.agg(F.min(var_cohorte).cast(IntegerType()).alias("Minimum cohorte"),
                                   F.max(var_cohorte).cast(IntegerType()).alias("Maximum cohorte"))\
            .toPandas()
        
        # Analysis of period (table)
        aux = table.select(*table.columns,(F.col(var_cohorte)/100).cast(IntegerType()).alias("Year"))
        periodo = aux.groupBy("Year").count().toPandas()
        periodo["Proportion"] = round(periodo["count"] / sum(periodo["count"]), 4)
        
        # Default rate table
        tabla_tm = aux.groupBy("Year").agg(F.count("Year").alias("Total"),
                                           F.sum(var_obj).alias("Bads")).toPandas()
        tabla_tm["Goods"] = tabla_tm["Total"] - tabla_tm["Bads"]
        tabla_tm["TM"] = round(tabla_tm["Bads"] / tabla_tm["Total"], 4)
        
        print("Universe analysis was completed.")

        return [dimension, rango_cohortes, periodo, tabla_tm]
    
    
    def balanc_sample(self, table, var_obj, train_size, seed):
        """
        Returns a stratified sample without replacement based on the fraction given on each stratum.
        """
        train = table.sampleBy(var_obj, fractions={0: train_size, 1: train_size}, seed=seed)
        test = table.subtract(train)
        train = train.select(*train.columns,(F.lit("train")).alias("sample"))
        test = test.select(*test.columns,(F.lit("test")).alias("sample"))
        table_all = train.union(test)

        return table_all
    
    
    def init_mature(self, table, params):
        """
        This function calculate the distribution of principals variables of interesting
        like: segment, sample and default.

        :param str table: Database to analyze
        :param str var_segm: segment column in database
        :param str var_sample: sample column in database
        :param str var_obj: var_obj in database, also called incmpl or bad
        :return: list with [distrib_seg, distrib_sample, distrib_segsam, distrib_client, distrib_samcli]
        
        Example:
            >>> w = init_mature(dummyBal, "segmento", "sample", "incmpl_d")
            >>> w[4].show()
            +------+------+-----+----------+
            |sample|incmpl|count|Proportion|
            +------+------+-----+----------+
            | train|     0|12578|    0.6289|
            |  test|     0| 5476|    0.2738|
            | train|     1| 1375|   0.06875|
            |  test|     1|  571|   0.02855|
            +------+------+-----+----------+
        """
        # Variables from dictionary
        var_segm = params["checkpoints2"]["var_segm"]
        var_sample = params["checkpoints2"]["var_sample"]
        var_obj = params["var_obj"]
        
        #---------------------------------------------
        # Segment distribution
        table = table.withColumnRenamed(var_segm, "segment")
        distrib_seg = table.groupBy("segment").count()
        distrib_seg = distrib_seg.select(*distrib_seg.columns,
                                         (F.col("count")/F.sum("count").over(Window.partitionBy())).alias("Proportion")) \
            .toPandas()

        #---------------------------------------------
        # Sample distribution
        table = table.withColumnRenamed(var_sample, "sample")
        distrib_sample = table.groupBy("sample").count()
        distrib_sample = distrib_sample.select(*distrib_sample.columns,
                                               (F.col("count")/F.sum("count").over(Window.partitionBy())).alias("Proportion")) \
            .toPandas()

        #---------------------------------------------
        # Segment and sample distribution
        distrib_segsam = table.groupBy("segment", "sample").count()
        distrib_segsam = distrib_segsam.select(*distrib_segsam.columns,
                                               (F.col("count")/F.sum("count").over(Window.partitionBy())).alias("Proportion")) \
            .orderBy(F.asc("segment"), F.desc("sample")) \
            .toPandas()

        #---------------------------------------------
        # Distribution by client
        table = table.withColumnRenamed(var_obj, "incmpl")
        distrib_client = table.groupBy("incmpl").count()
        distrib_client = distrib_client.select(*distrib_client.columns,
                                               (F.col("count")/F.sum("count").over(Window.partitionBy())).alias("Proportion")) \
            .orderBy(F.asc("incmpl")) \
            .toPandas()

        #---------------------------------------------
        # Distribution by sample and client
        distrib_samcli = table.groupBy("sample", "incmpl").count()
        distrib_samcli = distrib_samcli.select(*distrib_samcli.columns,
                                               (F.col("count")/F.sum("count").over(Window.partitionBy())).alias("Proportion")) \
            .orderBy(F.asc("incmpl"), F.desc("sample")) \
            .toPandas()
        
        print("Init mature analysis was completed.")

        Lst = [distrib_seg, distrib_sample, distrib_segsam, distrib_client, distrib_samcli]

        return Lst
    

    def evol_cohorte(self, table, params):
        """
        Evolution of default ratios through the differents cohortes on data base.
        
        :param str table: Database to analyze
        :param str var_cohorte: cohorte column in database
        :param str var_segm: segment column in database
        :param str var_obj: var_obj in database, also called incmpl or bad
        :return: table with columns cohorte, total, malos, buenos, tm
        """
        # Variables from dictionary
        var_segm = params["checkpoints2"]["var_segm"]
        var_cohorte = params["checkpoints2"]["var_cohorte"]
        var_obj = params["var_obj"]
        
        table = table.withColumnRenamed(var_cohorte, "cohorte")
        Lst = []
        
        if var_segm == None:
            evol_cohorte = table.groupBy("cohorte") \
                .agg(F.count(var_cohorte).alias("total"), F.sum(var_obj).alias("bads")) \
                .orderBy(F.asc("cohorte")).toPandas()
            evol_cohorte["goods"] = evol_cohorte["total"] - evol_cohorte["bads"]
            evol_cohorte["tm"] = round(evol_cohorte["bads"] / evol_cohorte["total"],4)
            
            Lst.append(evol_cohorte)
            
        else:
            max_iter = table.agg(F.max(F.col(var_segm)).alias("max")).toPandas()
            for i in range(1,max_iter["max"][0]+1):
                evol_cohorte = table.filter(F.col(var_segm) == i).groupBy("cohorte") \
                    .agg(F.count(var_cohorte).alias("total"), F.sum(var_obj).alias("bads")) \
                    .orderBy(F.asc("cohorte")).toPandas()
                evol_cohorte["goods"] = evol_cohorte["total"] - evol_cohorte["bads"]
                evol_cohorte["tm"] = round(evol_cohorte["bads"] / evol_cohorte["total"],4)
                evol_cohorte["segment"] = i

                # Save results
                Lst.append(evol_cohorte)
            
        print("Evolution cohorte analysis was completed.")

        return Lst
    
        
    def tmora_seg(self, table, params):
        """
        Summary table with columns sample, segment, total, bads, goods and default ratio
        
        :param str table: Database to analyze
        :param str var_segm: segment column in database
        :param str var_sample: sample column in database
        :param str var_obj: var_obj in database, also called incmpl or bad
        :return: summary table
        """
        # Variables from dictionary
        var_segm = params["checkpoints2"]["var_segm"]
        var_sample = params["checkpoints2"]["var_sample"]
        var_obj = params["var_obj"]
        
        seg_tmora = table.groupBy(var_sample,var_segm).agg(F.count(var_segm).alias("total"), F.sum(var_obj).alias("bads")) \
            .orderBy(F.asc(var_segm), F.desc(var_sample))
        seg_tmora = seg_tmora.select(*seg_tmora.columns,
                                     (F.col("total") - F.col("bads")).alias("goods"),
                                     F.round(F.col("bads") / F.col("total"), 5).alias("TM"))
        
        seg_tmora = seg_tmora.toPandas()
        print("tmora for each segment was competed.")

        return seg_tmora
    
    
    def indice_estab_percentil(self, table, params):
        """
        This function calculate Stability index between two samples of the spark
        data frame. The formula is:
        summation of (% sample x - % sample y) * ln(% sample x / % sample y)

        Decision Rules:
            - IE < 0.10 means doesn't exist differences
            - 0.10 <= IE (0.25 or 0.30) means exists some differences
            - IE > (0.25 or 0.30) means exists significative differences

        :param DataFrame table: Database to analyze
        :param list var_interes: list of variables to analyze
        :param str var_segm: segment column in database
        :param int cant_percent: Numbers of percentiles to consider in the analysis
        :return: list with [listValuesQ_all, table_xy_all, IE]

        Example:
            >>> var_interes = ["variable_1","variable_2","variable_3"]
            >>> a = indice_estab(spark_table, var_interes, "sample", 5)
            >>> a[0]
            [-inf, 0.012, 0.023, ..., inf]
        """
        # Variables from dictionary
        var_interes = params["checkpoints2"]["var_interes"]
        var_segm = params["checkpoints2"]["var_segm"]
        cant_percent = params["checkpoints2"]["cant_percent"]
        especial_values = params["special_values"]
        
        max_segm = table.select(var_segm).distinct().toPandas()
        max_segm = max_segm.sort_values(var_segm)
        segm_list = max_segm[var_segm].tolist()

        if len(max_segm) > 2:
            print("For IE, there are three or more labels in column '{}', reduce them in two...".format(var_segm))
        else:
            print("For IE, the segments to analize are:",segm_list)

            if type(max_segm.loc[0][0]) == str:
                seg1 = str(max_segm.loc[0][0])
                seg2 = str(max_segm.loc[1][0])
            else:
                seg1 = int(max_segm.loc[0][0])
                seg2 = int(max_segm.loc[1][0])
            # Empty list of percentils
            listValuesQ_all = []
            # Empty data frame
            table_xy_all = pd.DataFrame()

            for i in var_interes:
                esp_values = especial_values[i]
                quant_x = table.filter((~F.col(i).isin(esp_values)) & (F.col(var_segm) == seg1)).select(i)
                quant_y = table.filter((~F.col(i).isin(esp_values)) & (F.col(var_segm) == seg2)).select(i)
                total_x = quant_x.count()
                total_y = quant_y.count()
                # Create a list of percentils
                percentiles = list(np.arange(1/cant_percent, 1, 1/cant_percent))
                percentiles = [float(round(x,2)) for x in percentiles]
                # Values for the partitions:
                listValuesQ = (quant_x.filter(F.col(i).isNull() != True).approxQuantile(i, percentiles, 0.0))
                listValuesQ = [-float("inf")] + listValuesQ + [float("inf")]
                listValuesQ_all = listValuesQ_all + listValuesQ
                # Bucketizer transforms a column of continuous features to a column of feature buckets.
                bucketizer = Bucketizer(splits=listValuesQ, inputCol=i, outputCol= "Perc")
                # Bucketizer table X
                bucketedData = bucketizer.transform(quant_x.select(i))
                #print("Bucketizer output with {} buckets in first table".format(len(bucketizer.getSplits())-1))
                table_x = bucketedData.groupBy("Perc").count().orderBy(F.asc("Perc")) \
                    .withColumnRenamed('count', 'Frequency_X') \
                    .withColumn('Proportion_X', F.round(F.col('Frequency_X') / total_x, 3))
                try:
                    # Bucketizer table Y
                    bucketedData = bucketizer.transform(quant_y.select(i))
                    #print("Bucketizer output with {} buckets in second table".format(len(bucketizer.getSplits())-1))
                    table_y = bucketedData.groupBy("Perc").count().orderBy(F.asc("Perc")) \
                        .withColumnRenamed('count', 'Frequency_Y') \
                        .withColumn('Proportion_Y', F.round(F.col('Frequency_Y') / total_y, 3))
                    # Join table_x and table_y
                    table_xy = table_x.join(table_y, on="Perc", how="left_outer")
                    # Estability Index calculated
                    table_xy = table_xy.select(*table_xy.columns,
                                               ((F.col("Proportion_X") - F.col("Proportion_Y")) * \
                                                    F.log(F.col("Proportion_X") / F.col("Proportion_Y"))).alias("IE"),
                                               F.lit(i).alias("Variable")
                                              ).toPandas()
                    # Concat tables
                    table_xy_all = pd.concat([table_xy_all, table_xy], axis=0, ignore_index=True)
                    print("indice_estab analysis for",i,"ready!")
                    #print("Bucketizer output of variable '{0}' with {1} buckets".format(i, len(bucketizer.getSplits())-1))
                except Exception:
                    print("Error: "+i)

            IE = table_xy_all.groupby("Variable", as_index=False).agg({"IE":sum})
               
        print("IE analysis was completed.")
        
        return [listValuesQ_all, table_xy_all, IE]
    
    
    def psi_table_percentil(self, table, params):
        """
        Get table with following columns cohorte, Q, Freq, Total, Proportion and Variable.

        :param DataFrame table: Database in spark to analyze
        :param list var_interes: list of variables to analyze
        :param str var_cohorte: cohorte column in database
        :param str cohorte_base: cohorte like reference to start the analysis
        :param int cant_percent: Numbers of percentiles to consider in the analysis
        :return: list with [table_psi, var_cohorte, percentiles, listValuesQ, group_psi_all]

        Example:
            >>> var_interes = ["num_1","num_2","num_3"]
            >>> b = psi_table(spark_table, var_interes, "cohorte", 201501, 5)
            Cohorte base asigned is: 201501
            Variable 'num_1' bucketized
            Variable 'num_2' bucketized
            Variable 'num_3' bucketized
            >>> b[4].limit(3).show()
            +-------+---+----+-----+----------+--------+
            |cohorte|  Q|Freq|Total|Proportion|Variable|
            +-------+---+----+-----+----------+--------+
            | 201501|0.0| 165|  829|   0.19903|   num_1|
            | 201501|1.0| 166|  829|   0.20024|   num_1|
            | 201501|2.0| 166|  829|   0.20024|   num_1|
            +-------+---+----+-----+----------+--------+
        """
        # Variables from dictionary
        var_interes = params["checkpoints2"]["var_interes"]
        var_cohorte = params["checkpoints2"]["var_cohorte"]
        cohorte_base = params["checkpoints2"]["cohorte_base"]
        cant_percent = params["checkpoints2"]["cant_percent"]
        esp_values = params["checkpoints2"]["especial_values"]

        
        if cohorte_base == None:
            min_cohorte = dummyBal.groupBy().min(var_cohorte).collect()[0].asDict()["min("+var_cohorte+")"]
            print("For PSI table, the cohorte base calculated (min) is:", min_cohorte)
        else:
            min_cohorte = int(cohorte_base)
            print("For PSI table, the cohorte base asigned is:",min_cohorte)

        lista_psi = [var_cohorte] + var_interes
        table_psi = table.filter(F.col(var_cohorte) >= cohorte_base).select(lista_psi).na.drop(how = "any")
        # Create a list of percentils
        percentiles = list(np.arange(1/cant_percent, 1, 1/cant_percent))
        percentiles = [float(round(x,2)) for x in percentiles]
        # Empty data frame
        group_psi_all = pd.DataFrame()
        listValuesQ_all = []

        for i in var_interes:
            # Values for the partitions
            #listValuesQ = (table_psi.filter(F.col(var_cohorte).isNull() != True).approxQuantile(i, percentiles, 0.0))
            listValuesQ = (table_psi.filter((F.col(var_cohorte) == min_cohorte) & (~F.col(i).isin(esp_values[i]))) \
                .approxQuantile(i, percentiles, 0.0))
            listValuesQ = list(set(listValuesQ))
            listValuesQ.sort()
            listValuesQ = [-float("inf")] + listValuesQ + [float("inf")]
            try:
                # Bucketizer transforms a column of continuous features to a column of feature buckets.
                bucketizer = Bucketizer(splits=listValuesQ, inputCol=i, outputCol= "Q")
                # Bucketizer table X
                bucketedData = bucketizer.transform(table_psi)
                group_psi = bucketedData.groupBy(var_cohorte,"Q").agg((F.count(F.col("Q"))).alias("Freq")) \
                    .orderBy(F.asc(var_cohorte), F.asc("Q"))
                total_psi = bucketedData.groupBy(var_cohorte).agg((F.count(F.col(var_cohorte))).alias("Total"))
                group_psi = group_psi.join(total_psi, on=var_cohorte, how="left_outer")
                group_psi = group_psi.select(*group_psi.columns,
                                             (F.round(F.col("Freq")/F.col("Total"),5)).alias("Proportion"),
                                             (F.lit(i)).alias("Variable")).toPandas()
                #print("Variable '{}' bucketized".format(i))
                group_psi_all = pd.concat([group_psi_all, group_psi], axis=0, ignore_index=True)
                listValuesQ_all.append(listValuesQ)
                print("PSI for",i,"ready!")
            except Exception:
                print("Error: " + i)
        
        print("PSI table analysis was completed.")

        Lst = [var_cohorte, percentiles, listValuesQ_all, group_psi_all, var_interes]

        return Lst
    
    
    
    
    

    
    
    
    
    
    