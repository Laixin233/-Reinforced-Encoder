from ast import arg
import os
import sys
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import shutil
import time
from tqdm import trange
import statistics
import numpy as np
import pandas as pd
import torch
from task.TaskLoader import Opt
from task.TaskTuner import StocHyperTuner as HyperTuner
from task.dataset import de_scale
from task.util import os_makedirs, os_rmdirs, set_logger
from task.util import plot_xfit,plot_xfit_agent,plot_hError,scaler_inverse
# from models.seq2seq import 
# from models.statistical.naive import Naive
import importlib
from tqdm.std import tqdm
import math
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

class Task(Opt):
    def __init__(self, args):
        self.exp_module_path = importlib.import_module('data.{}.config.{}.H{}'.format(
            args.datafolder.replace('/', '.'), args.dataset,args.H))  # 引入配置
        self.save_model = args.save_model
        self.data_config(args)
        self.model_config(args)
        self.exp_config(args)
        self.data_subconfig()

    def data_config(self, args):
        self.data_name = args.dataset
        data_opts = getattr(self.exp_module_path, args.dataset + '_data') #H24的数据类
        self.data_opts = data_opts(args)

    def data_subconfig(self,):
        self.data_opts.arch = self.model_opts.arch
        self.data_opts.sub_config()#read data

    def model_config(self, args):
        self.model_name = args.model
        
        # load the specifical config firstly, if not exists, load the common config
        if hasattr(self.exp_module_path, self.model_name):
            model_opts = getattr(self.exp_module_path,args.model)
        else:
            try:
                share_module_path = importlib.import_module('data.base')
                model_opts = getattr(share_module_path, self.model_name + '_default')
            except:
                raise ValueError('Non-supported model {} in the data.base module, please check the module or the model name'.format(self.model_name))

        self.model_opts = model_opts()#RNNRL
        self.model_opts.hyper.merge(opts=self.data_opts.info)
        self.model_opts.hyper.H = args.H
        self.model_opts.tuner.name = args.tuner_name

        if self.model_opts.arch == 'cnn':
            if not self.model_name == 'clstm':
                self.model_opts.hyper.kernel_size = math.ceil(
                    self.model_opts.hyper.steps / 4)

    def model_import(self,):
        model = importlib.import_module(self.model_opts.import_path)
        model = getattr(model, self.model_opts.class_name)
        # model = model(self.model_opts.hyper, self.logger)  # transfer model in the conduct func.
        return model

    def exp_config(self, args):
        cuda_exist = torch.cuda.is_available()
        if cuda_exist and args.cuda:
            self.device = torch.device('cuda:{}'.format(args.gid))
        else:
            self.device = torch.device('cpu')

        if 'statistic' in vars(self.model_opts):
            self.device = torch.device('cpu')

        self.exp_dir = 'trial' if args.test == False else 'test'

        if args.mo is not None:
            self.exp_dir = os.path.join(self.exp_dir, args.mo)

        self.exp_dir = os.path.join(
            self.exp_dir, 'normal') if self.data_opts.info.normal else os.path.join(self.exp_dir, 'minmax')

        assert args.diff == False  # Not support differential preprocess yet!
        # self.exp_dir += '_diff'

        self.exp_dir = os.path.join(self.exp_dir, args.dataset)
        # arg.H = 24/48/96
        task_name = os.path.join('h{}'.format(
            arg.H),'{}.refit'.format(args.model)) if 'refit' in self.model_opts.hyper.dict and self.model_opts.hyper.refit else os.path.join('h{}'.format(args.H),'{}'.format(args.model) )

        self.task_dir = os.path.join(self.exp_dir, task_name)
        self.task_dir = os.path.join(self.task_dir, self.model_opts.hyper.component)
        if args.test and args.clean:
            os_rmdirs(self.task_dir)
        os_makedirs(self.task_dir)

        if args.test and args.logger_level != 20:
            self.logger_level = 50  # equal to critical
        else:
            self.logger_level = 20  # equal to info

        self.rep_times = args.rep_times



        self.model_opts.hyper.device = self.device
        self.tune = args.tune  # default False
        
        
        self.experiment = 'results'
        self.retain = os.path.join(self.experiment, args.dataset)        
        self.exp_root = os.path.join(self.retain, task_name)
        self.exp_task = os.path.join(self.exp_root, self.model_opts.hyper.component)
        os_makedirs(self.exp_task)
        

    def logger_config(self, dir, stage, cv, sub_count):
        log_path = os.path.join(dir, 'logs',
                                '{}.cv{}.series{}.log'.format(stage, cv, sub_count))
        log_name = '{}.series{}.cv{}.{}'.format(
            self.data_name, sub_count, cv, self.model_name)
        logger = set_logger(log_path, log_name, self.logger_level)
        return logger

    def conduct(self,):
        # init and mkdir taskdir
        # generate the subPack dataset
        if self.tune:
            self.tuning()

        for sub_count, series_Pack in enumerate(tqdm(self.data_opts.seriesPack)):
            # self.task_dir = os.path.join(self.task_dir, 'series_{}'.format(sub_count))
            assert sub_count == series_Pack.index
            
            self.series_dir = os.path.join(
                self.exp_task, 'series{}'.format(sub_count))
            self.measure_dir = os.path.join(self.series_dir, 'eval_results')
            os_makedirs(self.measure_dir)

            for i in trange(self.rep_times):
                result_file = os.path.join(
                    self.measure_dir, 'results_{}.series_{}.npy'.format(i, sub_count))

                os_makedirs(os.path.dirname(result_file))
                    
                if i > 0 and 'statistic' in self.model_opts.dict:
                    assert self.model_opts.statistic
                    result0 = str(os.path.join(
                        self.measure_dir, 'results_{}.series_{}.npy'.format(0, sub_count)))
                    shutil.copy(result0, result_file)
                    continue

                # loading the best paramters:
                cLogger = self.logger_config(
                    self.series_dir, 'train', i, sub_count)
                cLogger.critical('*'*80)
                cLogger.critical('Dataset: {}\t Model:{} \t InputL:{}\t H: {}\t Trail: {}'.format(
                    self.data_name, self.model_name, self.data_opts.info.steps,self.model_opts.hyper.H, i))
                
                with threadpool_limits(limits=8, user_api="blas"):
                    
                    self.conduct_iter(i, series_Pack, result_file, cLogger)

    def conduct_iter(self, i, subPack, result_file, clogger,_seed = None):

        try:
            if self.tune:
                best_hyper = self.load_tuning(subPack)
                self.model_opts.hyper.update(best_hyper)
                clogger.info("Updating tuning result complete.")
                clogger.critical('-'*80)

            if _seed is None:
                self.seed = i
            else:
                self.seed = _seed
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            clogger.critical(
                'For {}th-batch-trainingLoading, loading the sub-datasets {}'.format(i, subPack.index))
            clogger.critical('-'*80)

            # Attention! sub.H can be different among multiple series in some cases.
            self.model_opts.hyper.H = subPack.H
            self.model_opts.hyper.series_dir = self.series_dir
            self.model_opts.hyper.sid = subPack.index
            self.model_opts.hyper.cid = i
            self.model_opts.hyper.PreTrained_dir = self.exp_dir
            model = self.model_import()
            model = model(self.model_opts.hyper, clogger)#seq2seqRL
            
            # train the model
            fit_info = model.xfit(subPack.train_loader, subPack.valid_loader)
            # torch.save(model.model,'{}/{}_H{}.pkl'.format(self.exp_task,'model',self.model_opts.hyper.H))
            
            file = os.path.join(self.exp_task,  'loss_curve_cv{}.series{}.png'.format(i, subPack.index))

            if isinstance(fit_info,Opt): 
                self.plot_fitInfo(fit_info, subId=subPack.index,
                                cvId=i, flogger=clogger)

            # --- conduct testing
            print('Start testing...')
            history_x, tgt, pred, pred_actions = model.loader_pred(subPack.test_loader) #b,L,H
            
            np.save(result_file,(tgt, pred))
            # self.plot_example(history_x, tgt, pred, pred_actions)
        
            clogger.critical('-'*50)
            
            


        except:
            clogger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def tuning(self,):
        try:
            self.tune = True

            # check tune condition, if not satisfying, jump tune.
            if 'innerTuning' in self.model_opts.dict:
                if self.model_opts.innerTuning == True:
                    self.tune = False

            if self.tune:
                for sub_count, series_Pack in enumerate(tqdm(self.data_opts.seriesPack)):
                    assert sub_count == series_Pack.index

                    self.series_dir = os.path.join(
                        self.exp_task, 'series{}'.format(sub_count))

                    tuner_dir = os.path.join(self.series_dir, 'tuner/{}'.format(self.model_opts.tuner.name))
                    tuner_dir = os.path.abspath(tuner_dir)  #### 确保使用绝对路径
                    os_makedirs(tuner_dir)
                    tuner_path = os.path.join(
                        tuner_dir, 'series{}.best.pt'.format(sub_count))

                    self.model_opts.tuner.dir = tuner_dir

                    tLogger = self.logger_config(
                        tuner_dir, 'tuning', 'T', sub_count)
                    
                    self.model_opts.hyper.PreTrained_dir =  self.exp_task
                    if not os.path.exists(tuner_path):
                        series_tuner = HyperTuner(
                            self.model_opts, tLogger, series_Pack)
                        series_tuner.conduct()
                        best_hyper = series_tuner.best_config
                        tLogger.critical('-'*80)
                        tLogger.critical('Tuning complete.')
                        torch.save(best_hyper, tuner_path)
                    else:
                        best_hyper = torch.load(tuner_path)

                    # best_hyper is a dict type
                    for (arg, value) in best_hyper.items():
                        tLogger.info("Tuning Results:\t %s - %r", arg, value)
            return self.tune

        except Exception as e:
            raise ValueError(e)
            # raise ValueError(
            #     '{}\nGot an error on tuning.\n{}'.format('!'*50, '!'*50))
            
    def plot_instanceError(self,  result_file,tgt, pred ):
        
        np.save(result_file,(tgt, pred))
        # --- compute and save per-instance test losses (MSE and MAE) and plot distributions
        # tgt and pred are numpy arrays (N, ...)
        tgt_arr = np.asarray(tgt)
        pred_arr = np.asarray(pred)

        # try to align shapes if necessary
        if tgt_arr.shape != pred_arr.shape:
            pred_arr = pred_arr.reshape(tgt_arr.shape)

        # compute per-sample MSE and MAE across all non-batch axes
        if tgt_arr.ndim > 1:
            reduce_axes = tuple(range(1, tgt_arr.ndim))
        else:
            reduce_axes = (1,) if tgt_arr.ndim == 1 else ()

        # defensive: if there is no reduce axis (1D), compute element-wise
        if len(reduce_axes) == 0:
            per_sample_mse = (tgt_arr - pred_arr) ** 2
            per_sample_mae = np.abs(tgt_arr - pred_arr)
        else:
            per_sample_mse = np.mean((tgt_arr - pred_arr) ** 2, axis=reduce_axes)
            per_sample_mae = np.mean(np.abs(tgt_arr - pred_arr), axis=reduce_axes)

        # save in the same results folder where metrics (.xlsx) are written
        self.measure_task = os.path.join(self.exp_task, 'results')
        os_makedirs(self.measure_task)
        mse_file = os.path.join(self.measure_task, 'test_mse.npy')
        mae_file = os.path.join(self.measure_task, 'test_mae.npy')
        np.save(mse_file, per_sample_mse)
        np.save(mae_file, per_sample_mae)

        # plot distributions (histograms) and save
        f = plt.figure()
        plt.hist(per_sample_mse, bins=50, alpha=0.6, label='per-sample MSE')
        plt.hist(per_sample_mae, bins=50, alpha=0.6, label='per-sample MAE')
        plt.xlabel('Loss')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Test per-instance loss distribution')
        dist_path = os.path.join(self.measure_task, 'test_loss_dist.png')
        f.savefig(dist_path)
        plt.close()
            
            

                
    def plot_example(self,history_x, tgt, pred,actions):

        # 提取第一个样本的数据
        num = 1
        history_last = history_x[num, :, -1]  # (96,)
        tgt_0 = tgt[num, :]  # (24,)
        pred_0 = pred[num, :]  # (24,)
        action_x = actions[num, :, 0]  # (96,)
        action_y = actions[num, :, 1]  # (96,)
        action_h = actions[num, :, 2]  # (96,)

        # ==================== 2. 拼接数据并画出对比图（类似示例图） ====================
        # 拼接历史数据和目标/预测数据
        history_with_tgt = np.concatenate([history_last, tgt_0])  # (120,)
        history_with_pred = np.concatenate([history_last, pred_0])  # (120,)

        # 创建类似示例图的样式
        fig, ax = plt.subplots(figsize=(8,4))

        # 绘制Ground-True（历史+真实目标）
        x_axis = np.arange(len(history_with_tgt))
        ax.plot(x_axis, history_with_tgt, linewidth=2.5, color='#1f77b4', label='Ground-True', alpha=0.9)

        # 绘制历史+预测）
        ax.plot(x_axis, history_with_pred, linewidth=2.5, color='#ff7f0e', label='Prediction', alpha=0.9)

        # 添加垂直线标记历史和预测的分界点
        ax.axvline(x=len(history_last)-1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        # 设置图表样式
        ax.set_xlabel('Time Step', fontsize=13)
        ax.set_ylabel('Value', fontsize=13,)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, len(history_with_tgt))

        # 设置刻度
        ax.tick_params(labelsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(self.series_dir, 'comparison_plot.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()



        # Excel 1: History+GroundTruth 和 History+Prediction
        df_combined = pd.DataFrame({
            'Time_Step': np.arange(len(history_with_tgt)),
            'History+GroundTruth': history_with_tgt,
            'History+Prediction': history_with_pred
        })

        # 添加额外的sheet用于分离的数据
        with pd.ExcelWriter(os.path.join(self.series_dir, 'output_data.xlsx'), engine='openpyxl') as writer:
            # Sheet 1: 拼接的主要数据
            df_combined.to_excel(writer, sheet_name='Combined_Data', index=False)
            
            # Sheet 2: 历史数据
            df_history = pd.DataFrame({
                'Time_Step': np.arange(len(history_last)),
                'History_Value': history_last
            })
            df_history.to_excel(writer, sheet_name='History', index=False)
            
            # Sheet 3: 目标和预测
            df_target_pred = pd.DataFrame({
                'Time_Step': np.arange(len(tgt_0)),
                'Ground_Truth': tgt_0,
                'Prediction': pred_0
            })
            df_target_pred.to_excel(writer, sheet_name='Target_vs_Prediction', index=False)
            # Sheet 4: Actions Last Dimension
            df_actions = pd.DataFrame({
                'Time_Step': np.arange(len(action_x)),
                'Actions_x': action_x,
                'Actions_h': action_h,
                'Actions_y': action_y
            })
            df_actions.to_excel(writer,sheet_name='actions_data', index=False)




        
        
    def load_tuning(self, subPack):
        tuner_dir = os.path.join(self.series_dir, 'tuner/{}'.format(self.model_opts.tuner.name))
        #tuner_dir = 'file://' + os.path.abspath(os.path.join(self.series_dir, 'tuner/{}'.format(self.model_opts.tuner.name)))
        tuner_path = os.path.join(
            tuner_dir, 'series{}.best.pt'.format(subPack.index))
        best_hyper = torch.load(tuner_path)
        if not os.path.exists(tuner_path):
            raise ValueError(
                'Invalid tuner path: {}'.format(tuner_path))
        return best_hyper
    
    def plot_pred(self,true,pred):
        f = plt.figure()
        # number = true.shape[0]
        plt.plot(true[:,0], label='True')
        plt.plot(pred[:,0], label='Pred')
        f.savefig(os.path.join(self.series_dir, 'prediction.png'))
        plt.close()

    def plot_agentInfo(self,agentSelected_Percent,agentActions,subId, cvId, flogger):
        if agentSelected_Percent is not None:
            plot_dir = os.path.join(self.series_dir, 'figuresAgent_Test/cv{}'.format(cvId))
            os_makedirs(plot_dir)
            save_name = 'series{}'.format(subId)
            location= plot_dir
            num_samples = agentSelected_Percent.shape[0]
            x = np.arange(start=1, stop=num_samples + 1)
            f = plt.figure()
            plt.plot(x, agentSelected_Percent[:num_samples,0], label='Auto_P')
            plt.plot(x, agentSelected_Percent[:num_samples,1], label='MSVR_P')
            if agentSelected_Percent.shape[1] == 3:
                plt.plot(x, agentSelected_Percent[:num_samples,2], label='MLP_P')
            plt.xlabel('Horizon')
            plt.ylabel('Percent')
            plt.legend()
            f.savefig(os.path.join(location, save_name + 'Test'+'.xfit.png'))
            plt.close()
            np.save(os.path.join(location, save_name) + '.percent', (agentSelected_Percent))
            np.savetxt(os.path.join(location, save_name) + '.Action.txt',agentActions)
    
    def save_mse(self,fit_info):
        
        t_mse_df = pd.DataFrame({'Round': range(len(fit_info.v_mse_list)),
        'mse': [t.item() if isinstance(t, torch.Tensor) else t for t in fit_info.t_mse_list]})
        v_mse_df = pd.DataFrame({'Round': range(len(fit_info.v_mse_list)),
        'mse': [t.item() if isinstance(t, torch.Tensor) else t for t in fit_info.v_mse_list]})                

        t_mse_df.to_csv(os.path.join(self.exp_task, 'mse_train_list.csv'),index=False)
        v_mse_df.to_csv(os.path.join(self.exp_task, 'mse_test_list.csv'),index=False)

    
    # def plot_loss_curve(self,fit_info, save_path):
    
    #     plt.figure()
    #     plt.plot(fit_info.loss_list, label='train_loss')
    #     plt.plot(fit_info.vloss_list, label='val_loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('RMSE Loss')
    #     plt.title('Training and Validation Loss Curve')
    #     plt.legend()
    #     plt.savefig(save_path)
    #     plt.close()

    def plot_fitInfo(self, fit_info, subId, cvId, flogger):
        if 'loss_list' in fit_info.dict and 'vloss_list' in fit_info.dict:
            plot_dir = os.path.join(self.series_dir, 'figures')
            os_makedirs(plot_dir)

            plot_xfit(fit_info,'cv{}.series{}'.format(cvId, subId), plot_dir)
            flogger.critical('Ploting complete. Saving in {}'.format(plot_dir))
        
        if 'agentP' in fit_info.dict and 'v_agentP' in fit_info.dict:
            plot_dir = os.path.join(self.series_dir, 'figures_agent/cv{}'.format(cvId))
            os_makedirs(plot_dir)

            plot_xfit_agent(fit_info,'series{}'.format(subId), plot_dir)
            flogger.critical('Ploting Agent complete. Saving in {}'.format(plot_dir))
            
    def evaluation(self, metrics=['rmse'], force_update=True):
        try:
            self.metrics = metrics
            eval_list = []
            eLogger = set_logger(os.path.join(self.exp_task, 'eval.log'), '{}.H{}.{}'.format(
                self.data_name, self.data_opts.info.H, self.model_name.upper()), self.logger_level)
            
            
            for sub_count in range(self.data_opts.info.num_series):

                ser_eval = []
                self.series_dir = os.path.join(
                    self.exp_task, 'series{}'.format(sub_count))
                self.measure_dir = os.path.join(
                    self.series_dir, 'eval_results')
                os_makedirs(self.measure_dir)
                self.measure_task = os.path.join(self.exp_task, 'results')
                os_makedirs(self.measure_task)

                for i in range(self.rep_times):
                    metric_file = os.path.join(
                        self.measure_dir, 'metrics_{}.series_{}.npy'.format(i, sub_count))
                    test_file = os.path.join(
                        self.measure_task, 'metrics_{}.series_{}.xlsx'.format(i, sub_count)
                    )
                    if os.path.exists(metric_file) and force_update is False:
                        eval_results = np.load(metric_file)
                    else:
                        eval_results = self.eval_iter(
                            i, sub_count)

                    eLogger.critical('*'*80)
                    eLogger.critical('Dataset: {}\t Model: {}\t H: {}\tSeries-id: {}\t Trail-id: {}'.format(
                        self.data_name, self.model_name, self.data_opts.info.H, sub_count, i))
                    for _i, eval_name in enumerate(self.metrics):
                        eLogger.critical(
                            'Testing\t{}:\t{:.4g}'.format(eval_name, eval_results[0, _i]))
                    ser_eval.append(eval_results)
                    np.save(metric_file, eval_results)
                eval_list.append(ser_eval)
                eLogger.critical('-'*80)

            self.eval_info = Opt()
            self.eval_info.series = []
            # if self.rep_times > 1:
            for sub_count, ser_eval in enumerate(eval_list):
                eLogger.critical('='*80)
                eLogger.critical('Dataset: {}\t Model: {}\t H: {}\t Series-id: {} \t Trail-Nums: {}'.format(
                    self.data_name, self.model_name, self.data_opts.info.H, sub_count, self.rep_times))

                series_eval_dict = self.eval_list2dict(ser_eval)
                extracted_data = {key: [value['mean'], value['std']] for key, value in series_eval_dict.items()}
                need_data = pd.DataFrame.from_dict(extracted_data, orient='index', columns=['mean', 'std'])
                save_path = test_file
                need_data.to_excel(save_path) #保存测试集实验结果metric数据
                self.eval_info.series.append(series_eval_dict)
                for metric_name in self.metrics:
                    eLogger.critical('Testing {}\tMean:\t{:.4g}\tStd:\t{:.4g}'.format(
                        metric_name, series_eval_dict[metric_name]['mean'], series_eval_dict[metric_name]['std']))
            
            eLogger.critical('@'*80)
            eLogger.critical('Dataset: {}\t Model: {}\t Component: {}\t H: {}\t Series-Nums: {}\t Trail-Nums: {}'.format(
                self.data_name, self.model_name,self.model_opts.hyper.component, self.data_opts.info.H, self.data_opts.info.num_series, self.rep_times))

            all_eval_list = [item for series in eval_list for item in series]
            eval_return = self.eval_list2dict(all_eval_list)
            for metric_name in self.metrics:
                eLogger.critical('Testing {}\tMean:\t{:.4g}\tStd:\t{:.4g}'.format(
                    metric_name, eval_return[metric_name]['mean'], eval_return[metric_name]['std']))
            
            self.eval_info.all = eval_return
            
            
            return self.eval_info
        except:
            eLogger.exception(
                '{}\nGot an error on evaluation.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def eval_iter(self, i, sub_count):
        result_file = os.path.join(
            self.measure_dir, 'results_{}.series_{}.npy'.format(i, sub_count))
        _test_target, _pred = np.load(result_file)
        eval_results_len = len(self.metrics)
        eval_results = np.zeros((1, eval_results_len))
        for i, eval_name in enumerate(self.metrics):
            measure_path = importlib.import_module('task.metric')
            eval = getattr(measure_path, eval_name)
            # if self.data_name == 'mg':
            #     # eval_result = eval(_test_target, _pred, ith = 84)
            #     eval_result = eval(_test_target, _pred)
            # else:
            if eval_name == 'mase':
                # naivePred = self.get_naivePred(sub_count)
                eval_result = eval(_test_target, _pred, self.data_opts.seriesPack[sub_count].avgNaiveError)
            else:
                eval_result = eval(_test_target, _pred)
            eval_results[0, i] = eval_result
        return eval_results

    def get_naivePred(self, subcount):
        subPack = self.data_opts.seriesPack[subcount]
        testloader =subPack.test_loader
        
        tx = []
        for batch_x, _ in testloader:
            tx.append(batch_x)
        tx = torch.cat(tx, dim=0).detach().cpu().numpy()
        if len(tx.shape) == 3:
            tx = tx[:, 0, :]
        else:
            tx = np.expand_dims(tx[:,0],1)
        
        _tx = de_scale(subPack, tx, tag='input')
        
        _pred = _tx[:, -1]
        return _pred        
        
        

    def eval_list2dict(self, _eval_list):
        eval_data = np.concatenate(_eval_list, axis=0)

        eval_return = {}
        for i, metric_name in enumerate(self.metrics):
            i_data = eval_data[:, i].tolist()

            # if self.rep_times * self.data_opts.info.num_series > 1 and len(eval_data) > 1:
            if len(eval_data) > 1:
                mean = statistics.mean(i_data)
                std = statistics.stdev(i_data, mean)
            else:
                mean = i_data[0]
                std = 0

            # eLogger.critical('Testing {}\tMean:\t{:.4g}\tStd:\t{:.4g}'.format(
            #     metric_name, mean, std))

            eval_return[metric_name] = {}
            eval_return[metric_name]['mean'] = mean
            eval_return[metric_name]['std'] = std
            eval_return[metric_name]['raw'] = i_data
            
        return eval_return
