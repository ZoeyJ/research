# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:06:08 2015

@author: Zoey
"""

#######    number example box   ######
# record
SS1AS2_com_counter=[0,2,0,1,1,0,1,1,0,4,0,0,0,0,0,1,0,0] # example: a list of len 18
SS1AS2_opp_counter=[0,2,0,1,1,0,1,1,0,4,0,0,0,0,0,1,0,0]
SS1AS2_ind_list=18
# stage3 decision
act_ind_list=1
act_opp_list=[1,1,0,0,1,1,1,2,1,0,2]
num_new_entry=1 # will affect the first column in counter

SS1_ind=5
SS1_opp_count=[2,2,0,0,0]
temp_common_state=[2,2,0,0,1]
act2_ind=1
act2_opp_list=[0,1,2,0]
######################################
#checklist
# a. S1AS2_ind_list==3


##########    stage two   ##########

## 1. best response mapping
# best response mapping
Max_num_new_entry=0
sim_number1=20
starttime = datetime.datetime.now()
Best_response_mapping_early_FB_T2(Total_Initial_point,temp_common_state,scale,sim_number1,Max_num_new_entry)
endtime = datetime.datetime.now()
print "normal project running time"+str((endtime-starttime).total_seconds())

def Best_response_mapping_early_FB_T2(Total_Initial_point,temp_common_state,scale,sim_number1,Max_num_new_entry):    
    Pr_prior=numpy.transpose(numpy.random.dirichlet(numpy.ones(3),size=Type))
    cost_of_learning_context=0
    lam_UB=1/(c[1]+cost_of_learning_context)
    lam_LB=0
    distance=0.2 #set initial distance large, so while loop can be initialized
    distance_lam=0.2    
    iteration=0    
    while ((distance>1e-3 or (distance_lam>1e-6 and lam_UB>1e-6 and lam_LB<Max_num_new_entry and lam_UB-lam_LB>1e-3) ) and iteration<200 ):
        lam_prior=(lam_UB+lam_LB)/2
        Pr=numpy.zeros((3,Type))+numpy.nan
        U_individual={}
        for i in range(Type):
            
            temp_state_opponent=copy.copy(temp_common_state)
            if temp_common_state[i]>0:
                temp_state_individual=i+1
                temp_state_opponent[i]=temp_state_opponent[i]-1
                U=[]
                for temp_action_individual in range(3):
                    U.append(action_specific_U_earlyFB_SS1(temp_state_individual,temp_state_opponent,temp_action_individual,Pr_prior,lam_prior,sim_number1)*scale)
                expU=numpy.exp(U)
                Pr[:,i]=expU/sum(expU)   
                U_individual[str((temp_state_individual,temp_state_opponent))]=(Pr[0,i]*U[0]+Pr[1,i]*U[1]+Pr[2,i]*U[2])/scale
            else:Pr_prior[:,i]=numpy.nan    
        
        # for the new entrant
        state_individual_entrant=0
        action_new_entry=1
        Utility_new_entry=(action_specific_U_earlyFB_SS1(state_individual_entrant,temp_common_state,action_new_entry,Pr_prior,lam_prior,sim_number1)-cost_of_learning_context)*scale 
        if Utility_new_entry>0:
            lam_LB=lam_prior
        else:
            lam_UB=lam_prior
        gradience=Pr-Pr_prior
        where_are_NaNs = numpy.isnan(gradience)
        gradience[where_are_NaNs] = 0 # converting 'nan' to zero. when there is no one in that type
        Pr_prior=copy.copy(Pr_prior+gradience*0.75/numpy.sqrt(iteration/2+1))
        gradience_lam=Utility_new_entry
        distance_lam=numpy.sqrt(gradience_lam*gradience_lam)
        distance=numpy.nansum(numpy.nansum(gradience*gradience))
        iteration=iteration+1    

    if lam_LB>=Max_num_new_entry:
        lam_eq=Max_num_new_entry  
    elif lam_UB<=1e-6:
        lam_eq=0        
    elif distance_lam<=1e-6 or lam_UB-lam_LB<=1e-3:
        lam_eq=lam_prior    
    else:
        lam_eq="no result, out of iteration"
        print distance_lam,lam_LB,lam_UB,lam_prior
    return (Pr_prior,lam_eq,U_individual,iteration)   



## 2. action specific individual utiltiy
mu2_opp=numpy.transpose(numpy.random.dirichlet(numpy.ones(3),size=Type))
def action_specific_U_earlyFB_SS1(SS1_ind,SS1_opp_count,act2_ind,mu2_opp,lam,sim_number1):
    # act: pure action; mu: strategy, probability of playing each pure action
    # given opponent strategy, draw sim_number1 of action realization
    actionlist=simulate_action(SS1_opp_count,mu2_opp,sim_number1)
    # draw number of new entry
    num_new_entry_list=numpy.random.poisson(lam, size=sim_number1)
    Utility_individual=[]
    for i in range(sim_number1):
        act2_opp_list=actionlist[i]
        num_new_entry=num_new_entry_list[i]
        #print SS1AS2_ind_list,SS1AS2_opp_counter,act_ind_list,act_opp_list,num_new_entry
        temp_Utility_individual=IndU_T2_given_SS1_act2_NE(SS1_ind,SS1_opp_count,act2_ind,act2_opp_list,num_new_entry)
        Utility_individual.append(temp_Utility_individual)
    action_specific_utility=numpy.mean(Utility_individual)
    return action_specific_utility

## 3. everything specific individual utility 
def IndU_T2_given_SS1_act2_NE(SS1_ind,SS1_opp_count,act2_ind,act2_opp_list,num_new_entry): 
    #input: temp state and temp action; output: probability of next period state
    # if no new entry and ind is an new entry, then he gets zero utility
    if SS1_ind==0 and num_new_entry==0:
        U_ind=0
    else:
        # organize SS_ind and SS_opp
        if SS1_ind==0:
            num_new_entry=num_new_entry-1 # if new entrant, opponent's new entrant number minus 1.          
        # indented two more than needed
        if type(act2_opp_list) is not list: act2_opp_list=act2_opp_list.tolist()
        SS1_opp_list=Tool_state_num_state_list(SS1_opp_count)
        Num_opp_without_NE=sum(SS1_opp_count)#+num_new_entry
        SS1AS2_opp_list=[]
        for playeri in range(Num_opp_without_NE):
            SS1AS2_opp_list.append(SS1AS2_together(SS1_opp_list[playeri],act2_opp_list[playeri]))
        SS1AS2_opp_list=[2]*num_new_entry+SS1AS2_opp_list
        SS1AS2_ind=SS1AS2_together(SS1_ind,act2_ind)
        # get the common state SS1AS2
        SS1AS2_com_list=SS1AS2_opp_list+[SS1AS2_ind]
        # turn list into counter
        SS1AS2_com_counter=Tool_list_to_counter(SS1AS2_com_list,(Type+1)*actType)
        SS1AS2_opp_counter=Tool_list_to_counter(SS1AS2_opp_list,(Type+1)*actType)
        # get the individual utility out of the common state matrix
        (Total_Initial_point,Max_num_new_entry)=(1,100)
        Pr_prior,lam_eq,U_individual,iteration=Best_response_mapping_early_FB_T3(Total_Initial_point,SS1AS2_com_counter,scale,sim_number1,Max_num_new_entry)
        U_ind=U_individual[str((SS1AS2_ind,SS1AS2_opp_counter))]
        U_ind=U_ind-c[act2_ind]
    return U_ind








##########    stage three   ##########
## best responce mapping (stage 3)
# 1. given a mu_{-i}, lam, compute "action specific utility"
# 2. draw realized opponent action (act3^{-i}) and Num of new entry  (NE3) "simulate opp action"
# 3. compute utility U^i_3 given act3^{-i}, NE3, SS1 ,AS2, a^i_3 "everything specific utility"
## 
def T3_best_score(SS1,AS2,act3,p):

def SS1AS2_together(SS1,AS2):
    SS1AS2=SS1*3+AS2+1
    return SS1AS2


##0. best response mapping
def Best_response_mapping_early_FB_T3(Total_Initial_point,SS1AS2_com_counter,scale,sim_number1,Max_num_new_entry):    
    Pr_prior=numpy.transpose(numpy.random.dirichlet(numpy.ones(3),size=(1+Type)*actType)) # randomly setting an initial probability point
    cost_of_learning_context=0 
    lam_UB=1/(c[1]+cost_of_learning_context)
    lam_LB=0
    distance=0.2 #set initial distance large, so while loop can be initialized
    distance_lam=0.2    
    iteration=0    
    while ((distance>1e-3 or (distance_lam>1e-6 and lam_UB>1e-6 and lam_LB<Max_num_new_entry and lam_UB-lam_LB>1e-3) ) and iteration<200 ):
        lam_prior=(lam_UB+lam_LB)/2
        Pr=numpy.zeros((3,(1+Type)*actType))+numpy.nan
        U_individual={}
        for i in range((Type+1)*actType):
            SS1AS2_opp_counter=copy.copy(SS1AS2_com_counter)
            if SS1AS2_com_counter[i]>0:
                SS1AS2_ind_list=i+1
                SS1AS2_opp_counter[i]=SS1AS2_opp_counter[i]-1
                U=[]
                for act_ind_list in range(3):
                    U.append(action_specific_U_SS1AS2(SS1AS2_ind_list,SS1AS2_opp_counter,act_ind_list,Pr_prior,lam_prior,sim_number1)*scale)
                expU=numpy.exp(U)
                Pr[:,i]=expU/sum(expU)   
                U_individual[str((SS1AS2_ind_list,SS1AS2_opp_counter))]=(Pr[0,i]*U[0]+Pr[1,i]*U[1]+Pr[2,i]*U[2])/scale
            else:Pr_prior[:,i]=numpy.nan    
        # for the new entrant
        state_individual_entrant_list=1
        action_new_entry_list=1
        Utility_new_entry=(action_specific_U_SS1AS2(state_individual_entrant_list,SS1AS2_com_counter,action_new_entry_list,Pr_prior,lam_prior,sim_number1)-cost_of_learning_context)*scale
        if Utility_new_entry>0:
            lam_LB=lam_prior
        else:
            lam_UB=lam_prior
        gradience=Pr-Pr_prior
        where_are_NaNs = numpy.isnan(gradience)
        gradience[where_are_NaNs] = 0 # converting 'nan' to zero. when there is no one in that type
        Pr_prior=copy.copy(Pr_prior+gradience*0.75/numpy.sqrt(iteration/2+1))
        gradience_lam=Utility_new_entry
        distance_lam=numpy.sqrt(gradience_lam*gradience_lam)
        distance=numpy.nansum(numpy.nansum(gradience*gradience))
        iteration=iteration+1    

    if lam_LB>=Max_num_new_entry:
        lam_eq=Max_num_new_entry  
    elif lam_UB<=1e-6:
        lam_eq=0        
    elif distance_lam<=1e-6 or lam_UB-lam_LB<=1e-3:
        lam_eq=lam_prior    
    else:
        lam_eq="no result, out of iteration"
        print distance_lam,lam_LB,lam_UB,lam_prior
    return (Pr_prior,lam_eq,U_individual,iteration)
## 1. "simulate action given mu"
def simulate_action_given_mu(SS1AS2_opp_counter,action_probability,sim_number1):
    action={}
    for i in range((Type+1)*actType):
        action[i]=numpy.random.multinomial(SS1AS2_opp_counter[i], action_probability[:,i], size=sim_number1)
        actionbig=action[0]
    for i in range(1,(Type+1)*actType):
        actionbig=numpy.concatenate((actionbig,action[i]),axis=1)    
    action_list=[]
    for i in actionbig:
        temp_action_list=[]
        for temp_type in range((Type+1)*actType):
            temp_action_list+=[0]*i[temp_type*3+0]+[1]*i[temp_type*3+1]+[2]*i[temp_type*3+2]
        action_list.append(temp_action_list)
    action_list=numpy.asarray(action_list)
    return action_list

## 2."action specific U"
mu_opp=numpy.transpose(numpy.random.dirichlet(numpy.ones(3),size=(1+Type)*actType))
def action_specific_U_SS1AS2(SS1AS2_ind_list,SS1AS2_opp_counter,act_ind_list,mu_opp,lam,sim_number1):
    # act: pure action; mu: strategy, probability of playing each pure action
    # given opponent strategy, draw sim_number1 of action realization
    actionlist=simulate_action_given_mu(SS1AS2_opp_counter,mu_opp,sim_number1)
    # draw number of new entry
    num_new_entry_list=numpy.random.poisson(lam, size=sim_number1)
    Utility_individual=[]
    for i in range(sim_number1):
        act_opp_list=actionlist[i]
        num_new_entry=num_new_entry_list[i]
        #print SS1AS2_ind_list,SS1AS2_opp_counter,act_ind_list,act_opp_list,num_new_entry
        temp_Utility_individual=IndU_T3_given_SS1AS2_act3_NE(SS1AS2_ind_list,SS1AS2_opp_counter,act_ind_list,act_opp_list,num_new_entry)
        Utility_individual.append(temp_Utility_individual)
    action_specific_utility=numpy.mean(Utility_individual)
    return action_specific_utility

## 3. "everything specific utility"
def IndU_T3_given_SS1AS2_act3_NE(SS1AS2_ind_list,SS1AS2_opp_counter,act_ind_list,act_opp_list,num_new_entry): 
    if SS1AS2_ind_list==1 and num_new_entry==0:
        Utility=0  # if no new entry and ind is an new entry (SS1=0,AS2=0), then he gets zero utility
    elif SS1AS2_ind_list==3: 
        Utility=0
        print "no previous submission, cannot revise"
    else:
        # organize SS_ind and SS_opp
        if SS1AS2_ind_list==0:
            num_new_entry=num_new_entry-1 # if new entrant, opponent's new entrant number minus 1. 
        Num_opp=sum(SS1AS2_opp_counter)+num_new_entry
        SS1AS2_opp_list=Tool_SSAS_num_SSAS_list(SS1AS2_opp_counter) 
        if type(act_opp_list) is not list: 
            act_opp_list=act_opp_list.tolist()
        SS1_opp_list=[((score-1)/3) for score in SS1AS2_opp_list]
        AS2_opp_list=[numpy.mod(score+2,3) for score in SS1AS2_opp_list]
        act_opp_list.extend([1]*num_new_entry)
        SS1_opp_list.extend([0]*num_new_entry)    
        AS2_opp_list.extend([0]*num_new_entry)   
        # drawing opponent score one by one
        nextT_best_score_opponent=numpy.zeros((sim_number, Num_opp),int)    
        p=0   
        for temp_opponent_action in act_opp_list:
            SS1_opp_temp=SS1_opp_list[p]
            AS2_opp_temp=AS2_opp_list[p]
            nextT_best_score_opponent[:,p]=T3_best_score(SS1_opp_temp,AS2_opp_temp,temp_opponent_action,p)
            p=p+1
        # drawing individual's own score 
        next_best_score_individual=numpy.zeros(sim_number,int)
        SS1_ind=(SS1AS2_ind_list-1)/3  
        AS2_ind=numpy.mod(SS1AS2_ind_list+2,3)
        next_best_score_individual=T3_best_score(SS1_ind,AS2_ind,act_ind,p)
        # cbind individual's and opponent's best current score in the next stage, with individual's draw in the front column
        nextT_best_score=numpy.concatenate((next_best_score_individual.reshape(1,-1).T,nextT_best_score_opponent),axis=1)
        nextT_best_score=numpy.minimum(5,nextT_best_score)
        # compare who is the winner in every simulated case (sim_number1)
        Max_nextT_best_score=copy.copy(numpy.amax(nextT_best_score,axis=1).reshape(-1,1)+numpy.linspace(0,0,Num_opp+1))#take the maximum of each row. (for each draw, take the maximum among P players)
        equal=(Max_nextT_best_score==nextT_best_score) 
        Utility_without_cost_individual=(sum(equal[:,0]*((numpy.linspace(1.0,1.0,sim_number)/numpy.sum(equal, axis=1)))))/sim_number
        Utility=Utility_without_cost_individual-c[act_ind]
    return Utility
######################################
    
    
    
    
    
def T3_best_score(SS1,AS2,act3,p):
    nextT_best_score=numpy.zeros(sim_number,int)
    if SS1==0:
        if AS2==1:
            if act3==0:
                nextT_best_score=sim_S[:,p]
            elif act3==1:
                nextT_best_score=sim_SS[:,p]
            elif act3==2:
                nextT_best_score=sim_SS[:,p]## change this to a new entry and a revision
        elif AS2==2:
            print "no previous submission, cannot revise"
        else: #AS2_ind==0
            if act3==1: 
                next_best_score_individual=sim_S[:,p]
            else: 
                print "something wrong going on, act3:",act3,"  SS1:", SS1,"  AS2:",AS2,"  p:",p
    else:
        if act3==0 and AS2==0:
            nextT_best_score=numpy.zeros(sim_number,int)+SS1
        elif (act3==0 and AS2==1) or (act3==1 and AS2==0):
            nextT_best_score=numpy.maximum(sim_S[:,p],SS1)
        elif (act3==0 and AS2==2) or (act3==2 and AS2==0):
            nextT_best_score=numpy.maximum(0,sim_R[SS1][:,p])+SS1
        elif act3==1 and AS2==1:
            nextT_best_score=numpy.maximum(sim_SS[:,p],SS1)
        elif act3==1 and AS2==2:
            nextT_best_score=numpy.maximum(sim_S[:,p],SS1+numpy.maximum(0,sim_R[SS1][:,p]))
        elif act3==2 and AS2==1:
            nextT_best_score=sim_S_S_RnoFB[SS1][:,p]
        elif act3==2 and AS2==2:
            nextT_best_score=sim_S_RFB_RnoFB[SS1][:,p]
    return nextT_best_score