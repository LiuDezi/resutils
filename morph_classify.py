# classify KiDS galaxies as irregular and regular galaxies.
# randomn forest is adopted
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from astropy.table import Table
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys

class ClassifyMorph(object):
    """
    Classify KiDS galaxies as irregular and regular
    """
    def __init__(
        self, 
        train_set_name, 
        test_set_name, 
        out_set_name,
        train_set_path, 
        test_set_path,
        out_set_path
        ):
        self.train_set_name = train_set_name
        self.train_set_path = train_set_path
        self.test_set_name = test_set_name
        self.test_set_path = test_set_path
        self.out_set_name = out_set_name
        self.out_set_path = out_set_path

    def load_test_set(self):
        test_set_abs = os.path.join(self.test_set_path, self.test_set_name)
        test_set = Table.read(test_set_abs, format="fits")
        test_set = self.drop(test_set)
        
        # output
        parameter_name = ['MAG_GAAP_g', 'MAG_GAAP_r', 'MAG_GAAP_i', 'Gini',
                          'M20', 'Concentration', 'Asymmetry', 'Smoothness',
                          'bias_corrected_scalelength_pixels', 'bulge_fraction', 
                          'T_B']
        #test_set.index = list(range(len(test_set)))
        gr = test_set.MAG_GAAP_g - test_set.MAG_GAAP_r
        ri = test_set.MAG_GAAP_r - test_set.MAG_GAAP_i
        test_set = test_set[parameter_name]
        test_set['gr'] = gr
        test_set['ri'] = ri
        ntest = len(test_set)
        print(f'测试样本训练数为{ntest}')
        return test_set.values

    def load_train_set(self):
        train_set_abs = os.path.join(self.train_set_path, self.train_set_name)
        train_set = Table.read(train_set_abs, format="fits")
        train_set = train_set.to_pandas()
        train_label= train_set.irr
        
        # output
        parameter_name = ['MAG_GAAP_g', 'MAG_GAAP_r', 'MAG_GAAP_i', 'Gini', 
                          'M20', 'Concentration', 'Asymmetry', 'Smoothness', 
                          'bias_corrected_scalelength_pixels', 'bulge_fraction',
                          'T_B']
        #train_set.index = list(range(len(train_set)))
        gr = train_set.MAG_GAAP_g - train_set.MAG_GAAP_r
        ri = train_set.MAG_GAAP_r - train_set.MAG_GAAP_i
        train_set = train_set[parameter_name]
        train_set['gr'] = gr
        train_set['ri'] = ri
        ntrain = len(train_set)
        print(f'训练样本训练数为{ntrain}')
        return train_set.values, train_label

    def classify(self, train_set, test_set, label_set):
        clf = RandomForestClassifier(n_estimators=100)
        clf = clf.fit(train_set, label_set)
        outp = clf.predict(test_set)
        pro = clf.predict_proba(test_set)
        pro = pd.DataFrame(pro)
        pro1 = pro[pro[0]>=0.68]
        pro2 = pro[pro[1]>=0.68]
        pro3 = pd.concat([pro1,pro2],axis=0)
        ##select high precision sample
        pro3ind = pro3.index.values
        npro = len(np.where(outp[pro3ind]==0)[0])
        print(f'高置信度规则样本为{npro}')
        return outp, pro3ind
    
    def write_set(self, pred_out, pro3ind):
        test_set_abs = os.path.join(self.test_set_path, self.test_set_name)
        test_set = Table.read(test_set_abs, format="fits")
        test_set = self.drop(test_set)
        #test_set.index = list(range(len(test_set)))
        
        # output
        output_param = ['RAJ2000', 'DECJ2000', 'MAG_GAAP_g', 'MAG_GAAP_r', 'MAG_GAAP_i', 
                        'Gini', 'M20', 'Concentration', 'Asymmetry', 'Smoothness',
                        'bias_corrected_scalelength_pixels', 'bulge_fraction', 'T_B',
                        'e1', 'e2', 'weight']

        gr = test_set.MAG_GAAP_g - test_set.MAG_GAAP_r
        ri = test_set.MAG_GAAP_r - test_set.MAG_GAAP_i
        test_set = test_set[output_param]
        test_set["gr"] = gr
        test_set['ri'] = ri
        test_set['irr'] = pred_out
        test_set['irr1'] = -1
        test_set.loc[pro3ind,'irr1'] = pred_out[pro3ind]
        test_set_out = Table.from_pandas(test_set)

        out_set_abs = os.path.join(self.out_set_path, self.out_set_name)
        test_set_out.write(out_set_abs, format='fits', overwrite=True)
        return

    def stats_plot(self):
        out_set_abs = os.path.join(self.out_set_path, self.out_set_name)
        out_set = Table.read(out_set_abs, format="fits")
        out_set = self.drop(out_set)
        df1 = out_set[out_set.irr==0]
        df2 = out_set[out_set.irr!=0]

        plt.figure()
        linbins = 50
        colum1 = (df1['e1']**2 + df1['e2']**2)**0.5
        colum2 = (df2['e1']**2 + df2['e2']**2)**0.5
        weights1 = df1['weight']
        weights2 = df2['weight']
        plt.hist(colum1,bins=linbins, facecolor="white",fill=False, histtype='step',
                 edgecolor="black",linewidth=1,alpha=0.6,label='$Regular$',density=True,weights=weights1)
        plt.hist(colum2,bins=linbins, facecolor="white",fill=False, histtype='step',
                 edgecolor="red",linewidth=1,alpha=0.6,label='$Irregular$',density=True,weights=weights2)
        plt.xlabel('e')
        plt.ylabel("P")
        plt.legend(loc='upper right')
        plt.show()

        self.plot(df1, df2, "Gini", bin_limit=(0.2, 1))
        self.plot(df1, df2, "M20",  bin_limit=(-2.5, 0))
        self.plot(df1, df2, "Concentration", bin_limit=(0, 5))
        self.plot(df1, df2, "Asymmetry", bin_limit=(-1.5, 1.5))
        self.plot(df1, df2, "Smoothness", bin_limit=(-0.5, 0.5))
        self.plot(df1, df2, "bias_corrected_scalelength_pixels", bin_limit=(0, 10))
        self.plot(df1, df2, "bulge_fraction", bin_limit=(0, 1))
        self.plot(df1, df2, "MAG_GAAP_g")
        self.plot(df1, df2, "MAG_GAAP_r")
        self.plot(df1, df2, "MAG_GAAP_i")
        self.plot(df1, df2, "gr")
        self.plot(df1, df2, "ri")

    def drop(self, data):
        gid1_morph = (data["Gini"]!=-99) & (data["M20"]!=-99) & (data["Concentration"]!=-99)
        gid2_morph = (data["Asymmetry"]!=-99) & (data["Smoothness"]!=-99)
        gid_color = (data["MAG_GAAP_g"]!=99) & (data["MAG_GAAP_r"]!=99) & (data["MAG_GAAP_i"]!=99)
        gid = gid1_morph & gid2_morph & gid_color
        data_new = data[gid].to_pandas()
        return data_new

    def plot(self, df1, df2, para, nbin=50, bin_limit=None):
        plt.figure()
        if bin_limit is None:
            linbins = nbin
        else:
            low, up = bin_limit
            linbins = np.linspace(low, up, nbin)
        colum1 = df1[[para]]
        colum2 = df2[[para]]
        weights1 = np.ones_like(colum1)/len(colum1)
        weights2 = np.ones_like(colum2)/len(colum2)
        plt.hist(colum1,bins=linbins, facecolor="white",fill=False, histtype='step',
                 edgecolor="black",linewidth=1,alpha=0.6,label='$Regular$',density=False,weights=weights1)
        plt.hist(colum2,bins=linbins, facecolor="white",fill=False, histtype='step',
                 edgecolor="red",linewidth=1,alpha=0.6,label='$Irregular$',density=False,weights=weights2)
        plt.xlabel(para)
        plt.ylabel("density")
        plt.legend(loc='upper right')
        plt.show()

class ClassifyStat(object):
    def __init__(
        self,
        train_set_name,
        train_set_path,
        ):
        self.train_set_name = train_set_name
        self.train_set_path = train_set_path

    def stats_plot(self):
        train_set_abs = os.path.join(self.train_set_path, self.train_set_name)
        train_set = Table.read(train_set_abs, format="fits")
        train_set = train_set.to_pandas()
        train_label= (train_set.irr)

        # output
        parameter_name = ['MAG_GAAP_g', 'MAG_GAAP_r', 'MAG_GAAP_i', 'Gini',
                          'M20', 'Concentration', 'Asymmetry', 'Smoothness',
                          'bias_corrected_scalelength_pixels', 'bulge_fraction',
                          'T_B']
        #train_set.index = list(range(len(train_set)))
        gr = train_set.MAG_GAAP_g - train_set.MAG_GAAP_r
        ri = train_set.MAG_GAAP_r - train_set.MAG_GAAP_i
        train_set = train_set[parameter_name]
        train_set['gr'] = gr
        train_set['ri'] = ri
        train_values = train_set.values
        
        #normalise
        for i in range(len(train_values[0])): train_values[:,i] = self.normalize(train_values[:,i])
        feature_number=len(parameter_name) + 2
        
        circle_time, corr = 50, 0
        corr1 = np.zeros(circle_time)
        corr2 = np.zeros(circle_time)
        corr3 = np.zeros(circle_time)
        #feature_importances defination
        tree_feature_importances = np.zeros(feature_number)
        tree_feature_importances1 = np.zeros((circle_time,feature_number))

        dataindex = [i for i in range(len(train_values))]
        trainlen = int(len(train_values)/3)*2
        testlen = len(train_values)- trainlen
        #correct rate (circle_timetimes)
        for j in range(circle_time):
            #randomly select training sample
            #makesure train sample
            ranselect = np.random.choice(dataindex,size=trainlen,replace=False)
            ranselect = np.sort(ranselect)
            X = train_values[ranselect]
            Y = train_label[ranselect]
            #makesure train sample
            ranselect_other = dataindex
            for i in range(trainlen):
                ranselect_other = np.delete(ranselect_other, np.where(ranselect_other == ranselect[i]))
            ranselect_other = np.sort(ranselect_other)
            #prediction and check
            clf = RandomForestClassifier(n_estimators=50)
            clf = clf.fit(X, Y)
            outp = clf.predict(train_values[ranselect_other])
            outt = train_label[ranselect_other].values
            corr1[j], corr2[j], corr3[j] = self.corrate(len(outp),outt,outp)
            
            #check importance
            tree_feature_importances1[j] = clf.feature_importances_
            tree_feature_importances = np.array(clf.feature_importances_)+np.array(tree_feature_importances)
        tree_feature_importances = tree_feature_importances/circle_time
        print("all sample correct rate:",np.sum(corr1)/circle_time)
        print("regular sample correct rate:",np.sum(corr2)/circle_time)
        print("irregular sample correct rate:",np.sum(corr3)/circle_time)
       
        s, s1 = 0, 0
        for  i in range(circle_time):
            s += (corr1[i]-corr)**2/(circle_time)
            s1 += (tree_feature_importances1[i]-tree_feature_importances)**2/(circle_time)
        std_err=s1**0.5

        feature_names = ['MAG_GAAP_g','MAG_GAAP_r','MAG_GAAP_i','Gini','M20',
                         'Concentration','Asymmetry','Smoothness','scalelength',
                         'bulge_fraction','T_B','gr','ri']
        sorted_idx = tree_feature_importances.argsort()
        feature_namesid = []
        for i in range(feature_number):
            feature_namesid.append(feature_names[sorted_idx[i]])
        
        y_ticks = np.arange(0, len(feature_namesid))
        fig, ax = plt.subplots()
        ax.barh(y_ticks, tree_feature_importances[sorted_idx],xerr=std_err,error_kw = {'ecolor' : '0.2', 'capsize' :6 })
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(feature_namesid)
        ax.set_title("Random Forest Feature Importances (MDI)")
        fig.tight_layout()
        plt.savefig("")
        plt.show()

        # confusion matrix
        self.confusion_matrix(outt, outp)

        return

    def confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)

        # 绘制混淆矩阵
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        # 设置标签和刻度
        tick_marks = np.arange(len(np.unique(y_true)))
        plt.xticks(tick_marks, ['regular','irregular'])
        plt.yticks(tick_marks, ['regular','irregular'])


        # 添加标签到矩阵格子
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

        # 添加坐标轴标签
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        #plt.savefig('/home/lwk/Desktop/confusionmatric.pdf')
        # 显示图形
        plt.show()

    def normalize(self, arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if min_val<0: arr = arr-min_val
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_arr = (arr-min_val)/(max_val-min_val)
        return normalized_arr

    def corrate(self, tlen, true, predict):
        c, c1, c2 = 0, 0, 0
        for i in range(tlen):
            if true[i]==predict[i]:
                c=c+1
            if true[i]==0:
                if true[i]==predict[i]:
                    c1=c1+1
            if true[i]==1:
                if true[i]==predict[i]:
                    c2=c2+1
        return c/tlen, c1/len(np.where(true==0)[0]), c2/len(np.where(true==1)[0])


