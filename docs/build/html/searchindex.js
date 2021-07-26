Search.setIndex({docnames:["hydra","index","modules","quickstart","sesemi","sesemi.config","sesemi.models","sesemi.models.backbones","sesemi.models.heads","sesemi.schedulers","sesemi.tools","sesemi.trainer","sesemi.trainer.conf"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["hydra.rst","index.rst","modules.rst","quickstart.rst","sesemi.rst","sesemi.config.rst","sesemi.models.rst","sesemi.models.backbones.rst","sesemi.models.heads.rst","sesemi.schedulers.rst","sesemi.tools.rst","sesemi.trainer.rst","sesemi.trainer.conf.rst"],objects:{"":{sesemi:[4,0,0,"-"]},"sesemi.collation":{RotationTransformer:[4,1,1,""]},"sesemi.config":{resolvers:[5,0,0,"-"],structs:[5,0,0,"-"]},"sesemi.config.resolvers":{AttributeResolver:[5,1,1,""],SESEMIConfigAttributes:[5,1,1,""]},"sesemi.config.resolvers.SESEMIConfigAttributes":{iterations_per_epoch:[5,2,1,"id0"],max_iterations:[5,2,1,"id1"],num_gpus:[5,2,1,"id2"],num_nodes:[5,2,1,"id3"]},"sesemi.config.structs":{ClassifierConfig:[5,1,1,""],ClassifierHParams:[5,1,1,""],ClassifierModelConfig:[5,1,1,""],DataConfig:[5,1,1,""],DataLoaderConfig:[5,1,1,""],DatasetConfig:[5,1,1,""],LRSchedulerConfig:[5,1,1,""],LearnerConfig:[5,1,1,""],LossCallableConfig:[5,1,1,""],LossHeadConfig:[5,1,1,""],RunConfig:[5,1,1,""],RunMode:[5,1,1,""],SESEMIConfig:[5,1,1,""]},"sesemi.config.structs.ClassifierConfig":{hparams:[5,2,1,"id4"]},"sesemi.config.structs.ClassifierHParams":{lr_scheduler:[5,2,1,"id5"],model:[5,2,1,"id6"],num_classes:[5,2,1,"id7"],optimizer:[5,2,1,"id8"]},"sesemi.config.structs.ClassifierModelConfig":{backbone:[5,2,1,"id9"],regularization_loss_heads:[5,2,1,"id10"],supervised_loss:[5,2,1,"id11"]},"sesemi.config.structs.DataConfig":{test:[5,2,1,"id12"],train:[5,2,1,"id13"],val:[5,2,1,"id14"]},"sesemi.config.structs.DataLoaderConfig":{batch_sampler:[5,2,1,"id15"],batch_size:[5,2,1,"id16"],batch_size_per_gpu:[5,2,1,"id17"],collate_fn:[5,2,1,"id18"],dataset:[5,2,1,"id19"],drop_last:[5,2,1,"id20"],num_workers:[5,2,1,"id21"],pin_memory:[5,2,1,"id22"],sampler:[5,2,1,"id23"],shuffle:[5,2,1,"id24"],timeout:[5,2,1,"id25"],worker_init_fn:[5,2,1,"id26"]},"sesemi.config.structs.DatasetConfig":{image_transform:[5,2,1,"id27"],name:[5,2,1,"id28"],root:[5,2,1,"id29"],subset:[5,2,1,"id30"]},"sesemi.config.structs.LRSchedulerConfig":{frequency:[5,2,1,"id31"],interval:[5,2,1,"id32"],monitor:[5,2,1,"id33"],name:[5,2,1,"id34"],scheduler:[5,2,1,"id35"],strict:[5,2,1,"id36"]},"sesemi.config.structs.LossCallableConfig":{callable:[5,2,1,"id37"],reduction:[5,2,1,"id38"],scale_factor:[5,2,1,"id39"],scheduler:[5,2,1,"id40"]},"sesemi.config.structs.LossHeadConfig":{head:[5,2,1,"id41"],reduction:[5,2,1,"id42"],scale_factor:[5,2,1,"id43"],scheduler:[5,2,1,"id44"]},"sesemi.config.structs.RunConfig":{accelerator:[5,2,1,"id45"],batch_size_per_gpu:[5,2,1,"id46"],data_root:[5,2,1,"id47"],dir:[5,2,1,"id48"],gpus:[5,2,1,"id49"],id:[5,2,1,"id50"],mode:[5,2,1,"id51"],num_epochs:[5,2,1,"id52"],num_iterations:[5,2,1,"id53"],num_nodes:[5,2,1,"id54"],pretrained_checkpoint_path:[5,2,1,"id55"],resume_from_checkpoint:[5,2,1,"id56"],seed:[5,2,1,"id57"]},"sesemi.config.structs.RunMode":{FIT:[5,2,1,"id58"],TEST:[5,2,1,"id59"],VALIDATE:[5,2,1,"id60"]},"sesemi.config.structs.SESEMIConfig":{data:[5,2,1,"id61"],learner:[5,2,1,"id62"],run:[5,2,1,"id63"],trainer:[5,2,1,"id64"]},"sesemi.datamodules":{SESEMIDataModule:[4,1,1,""]},"sesemi.datamodules.SESEMIDataModule":{test:[4,2,1,""],test_dataloader:[4,3,1,""],train:[4,2,1,""],train_batch_sizes:[4,2,1,""],train_batch_sizes_per_gpu:[4,2,1,""],train_batch_sizes_per_iteration:[4,2,1,""],train_dataloader:[4,3,1,""],val:[4,2,1,""],val_dataloader:[4,3,1,""]},"sesemi.datasets":{dataset:[4,4,1,""],image_folder:[4,4,1,""],register_dataset:[4,4,1,""]},"sesemi.learners":{Classifier:[4,1,1,""]},"sesemi.learners.Classifier":{backbone:[4,5,1,""],configure_optimizers:[4,3,1,""],forward:[4,3,1,""],training_epoch_end:[4,3,1,""],training_step:[4,3,1,""],training_step_end:[4,3,1,""],validation_epoch_end:[4,3,1,""],validation_step:[4,3,1,""],validation_step_end:[4,3,1,""]},"sesemi.models":{backbones:[7,0,0,"-"],heads:[8,0,0,"-"]},"sesemi.models.backbones":{base:[7,0,0,"-"],timm:[7,0,0,"-"]},"sesemi.models.backbones.base":{Backbone:[7,1,1,""]},"sesemi.models.backbones.base.Backbone":{freeze:[7,3,1,""],out_features:[7,2,1,"id0"]},"sesemi.models.backbones.timm":{PyTorchImageModels:[7,1,1,""]},"sesemi.models.backbones.timm.PyTorchImageModels":{forward:[7,3,1,""],out_features:[7,2,1,""],training:[7,2,1,""]},"sesemi.models.heads":{loss:[8,0,0,"-"]},"sesemi.models.heads.loss":{LossHead:[8,1,1,""],RotationPredictionLossHead:[8,1,1,""]},"sesemi.models.heads.loss.LossHead":{build:[8,3,1,""],forward:[8,3,1,""],logger:[8,2,1,""],training:[8,2,1,""]},"sesemi.models.heads.loss.RotationPredictionLossHead":{build:[8,3,1,""],forward:[8,3,1,""],training:[8,2,1,""]},"sesemi.schedulers":{lr:[9,0,0,"-"],weight:[9,0,0,"-"]},"sesemi.schedulers.lr":{PolynomialLR:[9,1,1,""]},"sesemi.schedulers.lr.PolynomialLR":{get_lr:[9,3,1,""]},"sesemi.schedulers.weight":{SigmoidRampupScheduler:[9,1,1,""],WeightScheduler:[9,1,1,""]},"sesemi.tools":{inference:[10,0,0,"-"]},"sesemi.tools.inference":{Predictor:[10,1,1,""],predict:[10,4,1,""]},"sesemi.tools.inference.Predictor":{predict:[10,3,1,""]},"sesemi.trainer":{cli:[11,0,0,"-"],conf:[12,0,0,"-"]},"sesemi.trainer.cli":{open_sesemi:[11,4,1,""]},"sesemi.transforms":{GammaCorrection:[4,1,1,""],center_crop_transforms:[4,4,1,""],multi_crop_transforms:[4,4,1,""],train_transforms:[4,4,1,""]},"sesemi.utils":{assert_same_classes:[4,4,1,""],compute_num_gpus:[4,4,1,""],load_checkpoint:[4,4,1,""],reduce_tensor:[4,4,1,""],sigmoid_rampup:[4,4,1,""],validate_paths:[4,4,1,""]},sesemi:{collation:[4,0,0,"-"],config:[5,0,0,"-"],datamodules:[4,0,0,"-"],datasets:[4,0,0,"-"],learners:[4,0,0,"-"],models:[6,0,0,"-"],schedulers:[9,0,0,"-"],tools:[10,0,0,"-"],trainer:[11,0,0,"-"],transforms:[4,0,0,"-"],utils:[4,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"],"5":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function","5":"py:property"},terms:{"0":[1,3,4,5,7],"00028":4,"001":3,"01":4,"02":4,"02242":4,"1":[1,3,4,5,9,10],"10":[3,4],"100":3,"16":[3,4],"1610":4,"1704":4,"180":4,"1e":4,"2":[3,4],"2019":1,"224":4,"225":4,"229":4,"256":4,"270":4,"3":[1,4],"32":[],"4":3,"406":4,"42":3,"456":4,"485":4,"5":[3,4],"6":[1,4,5],"80":[1,3],"9":3,"90":[1,4],"91":1,"99":4,"case":[1,3,4],"class":[0,3,4,5,7,8,9,10],"default":[0,1,3,4,5],"do":4,"enum":5,"float":[4,5,7,9],"function":[3,4,7,11],"import":[0,3,4],"int":[3,4,5,7,8,9],"long":1,"new":8,"null":[3,5],"return":[3,4],"switch":4,"true":[3,4,5,7],"while":[1,4,7],A:[1,3,4,5,8,9],As:[3,8],At:4,But:4,For:[0,3,4,5],If:[1,4],In:[0,1,3,4],It:4,The:[0,1,3,4,5,7,8,9,11],Then:1,There:4,These:[0,3,5],To:[1,3],With:[1,4],_:1,_lrschedul:9,_target_:[3,5],ab:4,abov:[1,3,4],absolut:5,acc:4,acceler:[3,5],access:[1,3],accuraci:[1,4],across:1,actual:[3,4,8],ad:8,adam:4,add:[0,1,4],add_imag:4,addit:[3,4,5,8],addition:[0,3],advanc:[1,4],after:4,afterward:7,ahead:3,ai:1,algorithm:4,all:[1,3,4,5,7],along:4,also:[1,3,4,5],alter:8,although:[3,7],amazonaw:1,an:[1,3,4,5,8],ani:[1,3,4,5,8],annot:3,anyth:[3,4],api:[3,5],append:4,appli:[3,4,5],applic:1,approach:0,appus:1,ar:[0,1,3,4,5],arbitrari:[0,4],architectur:[1,6],arg:[1,3,4,10],argmax:4,argument:[0,3,4,8],arxiv:4,assert_same_class:4,assign:4,associ:4,assum:3,attribut:[3,5],attributeresolv:[3,5],author:1,automat:[0,4],avail:[3,5],averag:4,avg:[3,7],avoid:8,back:[3,4],backbon:[1,2,3,4,5,6,8],backend:4,background:3,backprop:4,backward:4,bar:[3,4],base:[1,2,3,4,5,6,8,9,10],bash:1,batch:[1,3,4,5,8],batch_cifar:4,batch_idx:4,batch_index:4,batch_mnist:4,batch_parts_output:4,batch_sampl:5,batch_siz:[4,5],batch_size_per_gpu:[3,5],becaus:4,been:4,befor:0,behavior:0,being:4,below:[1,3,4],benchmark:1,between:4,bicub:[],big:4,bind:1,bit:4,booktitl:1,bool:[4,5,7,8],boost:1,both:[1,3],box:[1,3],branch:3,brief:3,browser:1,build:[1,3,4,8],builder:[3,4],built:[0,1],c:1,calcul:4,call:[4,7],callabl:[3,4,5],callback:[3,5],can:[0,1,3,4,5,8],cannot:[3,5],care:7,cc:3,cd:[0,1],center:4,center_crop_transform:[3,4],certain:[3,5],chang:4,check:[0,3,4],checkpoint:[1,3,4,5],checkpoint_path:4,choic:1,choos:4,cifar:4,cifar_load:4,cite:1,ckpt:1,classif:7,classifi:[1,3,4,5],classifierconfig:[3,5],classifierhparam:[3,4,5],classifiermodelconfig:5,cleanli:3,cli:[0,2,3,4,5],clone:[],closur:4,cn:[0,1,3],code:[1,4],collat:[1,2,3,5],collate_fn:[3,5],collect:[3,4],com:1,come:0,comma:[3,4,5],command:[0,1,10],common:[3,5],competit:1,complex:3,compos:[0,3,4],composit:3,comput:[3,4,5,7,8],compute_num_gpu:4,concept:1,conf:[2,4,11],config:[0,2,3,4,11],configur:[0,1,4,5,11,12],configure_optim:4,consid:1,construct:[0,3],contain:4,contemporari:1,content:2,continu:4,contribut:1,control:4,convent:1,core:4,correct:4,correspond:4,cosineann:4,could:4,coupl:3,cours:[],cpu:[3,4,5],crash:4,creat:1,crop:4,crop_dim:4,cross:3,crossentropyloss:3,cuda:5,curl:1,curr_it:4,current:[1,3,4],custom:[3,4,5],cv:1,cycl:4,d:4,data:[0,1,3,4,5,8],data_root:[3,5],dataclass:3,dataconfig:[3,5],dataload:[4,5],dataloader_i_output:4,dataloader_idx:4,dataloader_out:4,dataloader_output_result:4,dataloaderconfig:[3,5],datamodul:[1,2],dataset:[1,2,5],datasetconfig:[3,5],ddp2:4,ddp:[3,4,5],decid:4,decod:4,decor:[3,4],deep:0,def:[3,4],defin:[0,3,4,7],definit:7,degre:4,demonstr:1,denomintaor:4,depend:1,describ:4,desir:0,detail:4,develop:3,deviat:4,dict:[3,4,5,8],dictionari:[3,4,5,8],didn:4,differ:[0,3,4],dim:4,dimens:4,dir:[3,5],directli:[3,5],directori:[0,1,3,5],dis_opt:4,dis_sch:4,disabl:4,displai:4,distribut:4,dive:0,doc:5,dockerfil:1,doe:5,don:[1,4],done:[1,3],doubl:8,download:[1,3,4],dp:[3,4,5],drop:5,drop_last:[3,5],drop_rat:[3,7],dst:1,dure:[1,3,5,8],dynam:0,e:[1,3,4,5],each:[3,4,5],earlier:[],easi:[0,1,3],easili:3,either:[3,4,5],en:[3,5],enabl:[0,1,3,4],encod:4,end:4,enforc:5,ensur:[1,4],enter:1,entri:4,entropi:3,environ:1,epoch:[1,3,4,5],eras:4,essenti:3,etc:[1,3,5],eval:4,evalu:1,everi:[4,7],exampl:[0,1,4,5],example_imag:4,exist:[4,5],expand:1,experi:[1,4],explor:1,exponenti:4,exponentiallr:4,expos:[3,5],extern:[0,3],extract:1,factor:5,fals:[3,4,5,7],familiar:0,fanci:1,fancier:4,fast:1,fastai:1,featur:[3,7,8],few:1,file:[0,3,5],filesystem:4,fill:3,final_metr:4,final_valu:4,find:[1,3],first:[3,4],fit:[3,4,5],flexibl:[0,3],flip:4,flyreelai:1,folder:[3,4],follow:[1,3,4,5],foo:3,form:3,format:3,former:[4,7],forward:[3,4,7,8],found:4,freez:[3,7],frequenc:[4,5],from:[0,1,3,4,5,7],full:[3,4,5],fulli:3,g:[1,3,4,5],gamma:4,gamma_rang:4,gammacorrect:4,gan:4,gen_opt:4,gen_sch:4,gener:[4,5,7],get:[4,5],get_lr:9,git:1,github:1,given:[0,4],global_pool:[3,7],go:[1,3],goal:1,goe:4,gpu:[1,3,4,5],gpu_0_pr:4,gpu_1_pr:4,gpu_n_pr:4,gradient:4,green:1,grid:4,group:[0,3,5],guid:4,h:3,ha:4,handl:4,happen:4,hardwar:4,have:[1,3,4],head:[2,3,4,5,6],help:3,here:[1,4],hidden:4,high:[0,1],highlight:5,home:1,hook:[4,7],horiziont:4,host:1,how:[3,4],howev:[1,3,4],hparam:[3,4,5],html:[3,5],http:[1,3,4,5],hydra:[1,3,12],hyper:1,hyperparamet:[3,5],id:[1,3,4,5],identifi:[3,5],ignor:7,imag:[3,4,5,7],image_fold:[3,4],image_transform:[3,4,5],imagecla:1,imagefold:4,imagenet:1,imagenett:1,imagewoof2:[1,3],imagewoof:[1,3],imagewoof_run01:[],imagewoof_test01:[],implement:4,improv:[1,4],includ:4,incorpor:1,increas:4,index:[1,4],indic:5,individu:4,infer:[2,4],inform:3,initi:[3,5],inject:0,inner:4,inproceed:1,input:[3,4],input_backbon:[3,8],input_data:[3,8],insid:4,inspect:3,instal:3,instanc:[3,4,7],instanti:[0,3,5],instead:[1,7],instruct:1,integ:[3,4,5],integr:1,interchang:0,interest:4,interfac:[3,7,8,9],intern:[0,3],interpol:[3,4],interpolationmod:[],interv:[4,5],invok:5,involv:0,io:[3,5],ipc:1,item:4,iter:[3,4,5],iterabledataset:[3,4],iterations_per_epoch:[3,5],iters_per_epoch:[3,9],its:4,kei:[0,4],keyword:4,kind:[3,5],know:[1,4],known:3,kwarg:[3,4,8],label:1,labels_hat:4,larg:[1,4],last:[1,4,5],last_epoch:9,later:4,latest:[1,3,5],latter:[4,7],lbfg:4,leaderboard:1,learn:[1,4,5,9],learner:[1,2,3,5],learnerconfig:[3,5],learningratemonitor:4,least:1,len:4,let:1,level:[0,4],librari:[1,3],light:1,lightn:[1,3,4,5,8],lightning_log:1,lightningdatamodul:4,lightningloggerbas:8,lightningmodul:4,like:[1,4],limit:1,line:10,list:[0,1,3,4,5],littl:1,load:[3,4,5],load_checkpoint:4,loader:[3,4,5],loader_a:4,loader_b:4,loader_n:4,log:[1,3,4,5],log_dict:4,logdir:1,logger:[4,8],look:[0,3],lookup:5,loss:[1,2,3,4,5,6],losscallableconfig:5,losshead:8,lossheadconfig:5,lowercas:4,lr:[2,3,4],lr_dict:4,lr_pow:[3,9],lr_schedul:[3,4,5,9],lrschedulerconfig:5,lstm:4,m:[],machin:1,mai:[3,8],main:[0,3,4,5,11],make:[0,3],make_grid:4,manag:[0,3],mani:1,map:[3,5],master:1,match:4,max:3,max_it:[3,9],max_iter:[3,5],maximum:[3,5],mean:[3,4,5],mechan:3,mention:4,method:[1,4,5],metric:[4,5],might:4,miniconda:1,miss:3,mkdir:1,ml:1,mnist:4,mnist_load:4,mode:[1,3,4,5],model:[1,2,3,4,5],model_di:4,model_gen:4,model_path:10,modelcheckpoint:3,modern:1,modul:[1,2,3],moduledict:8,momentum:3,monitor:[1,3,4,5],more:4,most:[1,4,5],mount:1,multi:4,multi_crop_transform:4,multi_gpu:4,multipl:[0,4,5],must:4,n:4,n_critic:4,name:[1,3,4,5,7],nce:4,nce_loss:4,ncrop:10,necessari:[1,4],need:[1,4,7],nest:[3,4],nesterov:3,neurip:1,next:4,nn:[3,4,7,8],node:[3,5],noisi:1,none:[3,4,5,8,9],norm:4,normal:4,note:[1,3],now:4,num_class:[3,5],num_crop:4,num_epoch:[3,5],num_gpu:[3,4,5],num_iter:[3,5],num_nod:[3,5],num_work:[3,5],number:[3,4,5,7,8],nutshel:0,nvidia:1,object:[0,3,4,5,8,9,10],obtain:1,often:4,omegaconf:[3,5],one:[1,3,4,7],onli:[1,4],open:1,open_sesemi:[0,1,3,11],oper:4,optim:[3,4,5,9],optimizer_idx:4,optimizer_on:4,optimizer_step:4,optimizer_two:4,option:[1,3,4,5,8,9],order:4,org:[1,4,5],os:1,other:[3,4,8],otherwis:3,our:1,out:[1,3,4],out_featur:7,outer:4,output:[3,4,5,7],over:4,overrid:[3,4],overridden:[4,7],overview:0,own:[0,3,4],p_eras:4,p_hflip:4,packag:[0,1,2,3],page:[1,3,4],paper:1,paradigm:1,paramet:[1,3,4,5,7,8,11],paramref:4,parent:[3,5],pars:3,part:[3,4],particular:3,pass:[3,4,5,7],path:[0,3,4,5],pattern:4,per:[3,4,5],perform:[1,5,7],phi:1,pin:5,pin_memori:[3,5],pip:3,placehold:8,pleas:[1,3,4],polynomi:9,polynomiallr:[3,9],popular:1,portion:4,possibl:4,power:3,practition:1,precis:4,pred:4,predict:[3,4,8,10],predictor:10,prefer:1,prepar:4,prepare_data:4,present:4,pretrain:[1,3,5,7],pretrained_checkpoint_path:[1,3,5],previou:4,primer:[1,3],principl:1,probabl:4,procedur:4,process:[1,4,5],produc:4,progress:4,project:[],propag:4,properti:4,protocol:1,provid:0,pseudocod:4,put:4,pwd:1,py:[],pypi:1,python:[0,1,3],pytorch:[1,3,4,5,7,8],pytorch_lightn:[3,4,8],pytorchimagemodel:[3,7],quickstart:1,r:[],ramp:[4,9],rampup:4,rampup_it:4,random:[3,4,5],random_resized_crop:4,rang:4,rate:[1,4,5,9],re:1,readthedoc:[3,5],realist:1,reason:3,recip:7,recommend:4,reduc:4,reduce_tensor:4,reducelronplateau:4,reduct:[3,4,5],refer:[3,5],referenc:[3,5],regist:[3,4,7],register_dataset:[3,4],registri:[4,5],regular:3,regularization_loss_head:[3,5],rel:[3,5],relat:1,relev:1,reload_dataloaders_every_epoch:4,repo:1,repositori:[1,7],requir:[1,4],resiz:4,resnet50d:[3,7],resolv:[2,3,4],respect:5,rest:4,restor:[3,5],result:4,resume_from_checkpoint:[3,5],rich:1,rm:1,robust:1,root:[1,3,4,5],rotat:[3,4,8],rotation_predict:[3,8],rotationpredictionlosshead:[3,8],rotationtransform:[3,4],run:[0,1,3,4,5,7],runconfig:[3,5],runmod:[3,5],runtim:[0,3],rwightman:7,s3:1,s:[1,3,4,5,7,11],same:4,sampl:[],sample_img:4,sampler:[4,5],save:8,save_last:3,save_top_k:3,scale:[4,5],scale_factor:[3,5],schedul:[2,3,4,5],schema:0,search:[0,1],second:4,section:[0,1,3],see:[1,3,4,5],seed:[3,5],select:0,self:4,semi:1,separ:[3,4,5],sequenti:4,sesemi:[0,3],sesemi_config:3,sesemi_imag:1,sesemiconfig:[3,5,11],sesemiconfigattribut:[3,5],sesemidatamodul:4,set:[0,1,3,4,5],setup:4,sgd:[3,4],share:8,shot:1,should:[1,3,4,5,7,8],shown:[3,4],shuffl:[3,4,5],sigmoid:[4,9],sigmoid_rampup:4,sigmoidrampupschedul:9,silent:7,similar:[3,4],simpli:1,sinc:7,singl:4,size:[1,3,4,5],skip:4,smooth:4,so:4,softmax:4,some:[0,3,4],someth:4,somewhat:3,sourc:[1,3],specif:[0,3,4,5],specifi:[3,4,5],split:4,split_batches_for_dp:4,src:1,standard:[1,4],star:1,state:[3,4,5],statist:1,step:[4,5,8],still:4,stop_rampup:9,store:[3,5],str:[3,4,5,7,8],strict:[4,5],string:5,struct:[2,3,4,11],structur:[0,1,5],sub_batch:4,subclass:7,subject:3,submodul:[1,2,6],subpackag:[1,2],subset:[3,4,5],sum:[4,5],supervis:[3,4],supervised_loss:[3,5],support:[0,1,3,4,5],syntax:[3,5],system:[0,3],t:[1,4],t_max:4,take:[1,7],taken:5,tar:1,task:[1,4],tell:4,tensor:[4,5,8],tensorboard:1,tensorflow:1,test:[3,4,5],test_dataload:4,test_step:4,text:[3,4],tgz:1,them:[1,7],thi:[0,1,3,4,5,7,8],thing:4,those:[0,3,4],through:[0,1,3,4],time:[1,3,4],timeout:5,timm:[1,2,4,6],titl:1,too:0,tool:[1,2,4],top1:3,top:4,topk:10,torch:[3,4,5,7,8,9],torchvis:[3,4],totensor:4,tpu:4,track:8,train:[1,3,4,5,7,8,11,12],train_batch:4,train_batch_s:4,train_batch_sizes_per_gpu:4,train_batch_sizes_per_iter:4,train_data:4,train_dataload:4,train_out:4,train_transform:[3,4],trainer:[2,3,4,5],training_epoch_end:4,training_step:4,training_step_end:4,training_step_output:4,tran:1,transesemi:1,transform:[1,2,3,5],trick:1,truncat:4,truncated_bptt_step:4,tune:1,tupl:4,turn:3,tutori:[],two:4,txt:[],type:[0,1,3,4,5,7],typic:0,u:1,underli:[0,3,4],unevenli:5,union:[3,4],unit:4,unlabel:1,unless:4,unsupervis:3,up:[4,9],updat:[4,5],us:[0,1,3,4,5],user:[0,1,3],user_id:1,util:[1,2,5],val:[3,4,5],val_acc:4,val_batch:4,val_data:4,val_dataload:4,val_loss:4,val_out:4,val_step_output:4,valid:[1,3,4,5],validate_path:4,validation_epoch_end:4,validation_step:4,validation_step_end:4,valu:[3,4,5],variabl:3,variou:1,veri:1,version:1,via:0,view:[1,3],virtual:1,vu:1,wai:0,warmup:[],warmup_epoch:[3,9],warmup_it:9,warmup_lr:[3,9],wasserstein:4,we:3,weheth:5,weight:[1,2,3,4,5],weight_decai:3,weightschedul:9,welcom:1,well:[0,3],what:4,whatev:4,when:[3,4,5],where:4,whether:[4,5],which:[0,1,3,4,8],whose:4,why:3,wish:4,within:7,without:[1,4],won:4,work:[0,1,3],worker:5,worker_init_fn:5,workshop:1,would:1,written:1,www:1,x:[4,7,10],xzv:1,y:4,yaml:[0,3],year:1,you:[1,3,4],your:[1,3,4],yourself:4,z:4},titles:["A Primer on Hydra","Image Classification with Self-Supervised Regularization","sesemi","Quickstart","sesemi package","sesemi.config package","sesemi.models package","sesemi.models.backbones package","sesemi.models.heads package","sesemi.schedulers package","sesemi.tools package","sesemi.trainer package","sesemi.trainer.conf package"],titleterms:{A:0,applic:0,backbon:7,base:7,built:3,citat:1,classif:1,cli:11,collat:4,concept:0,conf:12,config:5,configur:3,content:[1,4,5,6,7,8,9,10,11,12],cours:[],crash:[],datamodul:4,dataset:[3,4],docker:1,exampl:3,featur:1,get:1,head:8,highlight:1,hydra:0,imag:1,indic:1,infer:10,instal:1,learner:4,loss:8,lr:9,manag:[],model:[6,7,8],modul:[4,5,6,7,8,9,10,11,12],packag:[4,5,6,7,8,9,10,11,12],pip:1,primer:0,quickstart:3,regular:1,resolv:5,sampl:[],schedul:9,self:1,sesemi:[1,2,4,5,6,7,8,9,10,11,12],start:1,struct:5,structur:3,submodul:[4,5,7,8,9,10,11],subpackag:[4,6,11],supervis:1,tabl:1,timm:7,tool:10,trainer:[11,12],transform:4,util:4,weight:9,why:1}})