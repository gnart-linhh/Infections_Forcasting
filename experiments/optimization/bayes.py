"""Homoscedastic Gaussian-process/EI minimizer with resumable RNG state."""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import qmc

NAMES = ('num_curves', 'k', 'closest')

#tính Expected Improvement (EI) cho các ứng viên
def expected_improvement(mean, std, best):
    mean, std = np.broadcast_arrays(np.asarray(mean, float), np.asarray(std, float))
    delta = best - mean
    z = np.divide(delta, std, out=np.zeros_like(delta), where=std>1e-12)
    result = delta * ndtr(z) + std * np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    return np.where(std>1e-12, np.maximum(result,0), np.maximum(delta,0))


class GaussianProcess:
    def __init__(self, x, y):
        self.x = np.asarray(x,float)
        y = np.asarray(y,float)
        self.mean = float(y.mean()); self.scale = max(float(y.std()),1e-9)
        self.y = (y-self.mean)/self.scale
        self.result = None
        bounds = [(-4,2)]*self.x.shape[1] + [(-12,0)] #khai báo giới hạn siêu tham số của GP

        #tính log-likelihood của GP và tối ưu hóa nó để tìm siêu tham số tốt nhất
        def likelihood(theta):
            try:
                factor = self.factor(theta)
                return float(.5*self.y@cho_solve(factor,self.y) +
                             np.log(np.diag(factor[0])).sum() + .5*len(y)*np.log(2*np.pi))
            except np.linalg.LinAlgError:
                return 1e30
        #Tối ưu siêu tham số bằng L-BFGS-B
        fits = [minimize(likelihood,np.r_[np.full(self.x.shape[1],length),noise],
                         bounds=bounds,method='L-BFGS-B') for length,noise in [(-1.,-6.),(0.,-2.)]]
        self.result = min(fits,key=lambda r:r.fun) #Chọn kết quả tốt nhất trong hai lần chạy
        if not np.isfinite(self.result.fun): #Kiểm tra kết quả
            raise RuntimeError('GP fit failed')
        #Lưu siêu tham số và xây dựng mô hình GP
        self.theta = self.result.x
        self.cholesky = self.factor(self.theta)
        self.alpha = cho_solve(self.cholesky,self.y)

    #Đo mức độ tương đồng giữa hai điểm
    def kernel(self,a,b,theta):
        delta = (np.asarray(a)[:,None,:]-np.asarray(b)[None,:,:])/np.exp(theta[:-1])
        return np.exp(-.5*np.sum(delta*delta,axis=2))

    #Tạo ma trận hiệp phương sai
    def factor(self,theta):
        covariance = self.kernel(self.x,self.x,theta)
        covariance += (np.exp(theta[-1])+1e-8)*np.eye(len(self.x))
        return cho_factor(covariance,lower=True)

    #Dự đoán MAE và độ bất định
    def predict(self,x):
        covariance = self.kernel(np.atleast_2d(x),self.x,self.theta) #Tính covariance giữa điểm mới và dữ liệu huấn luyện
        mean = self.mean+self.scale*(covariance@self.alpha) #Dự đoán giá trị trung bình
        variance = np.maximum(0,1-np.sum(covariance*cho_solve(self.cholesky,covariance.T).T,axis=1)) #Dự đoán phương sai
        return mean, self.scale*np.sqrt(variance) #Dự đoán độ lệch chuẩn

#Chuyển điểm liên tục thành tham số nguyên
def decode(unit,bounds):
    bounds = np.asarray(bounds,float)
    return np.clip(np.ceil(bounds[:,0]+np.asarray(unit)*(bounds[:,1]-bounds[:,0])),bounds[:,0],bounds[:,1]).astype(int)

#Đề xuất bộ tham số tiếp theo bằng EI
def propose(history,bounds,rng,candidates=1024):
    bounds = np.asarray(bounds,float)
    x = np.array([[row[n] for n in NAMES] for row in history])
    y = np.array([row['validation_mae'] for row in history]) 
    unit = (x-bounds[:,0])/(bounds[:,1]-bounds[:,0]) #Chuẩn hóa không gian đầu vào
    model = GaussianProcess(unit,y) #Huấn luyện Gaussian Process
    pool = qmc.LatinHypercube(d=len(bounds),rng=rng).random(candidates) #Tạo các ứng viên bằng phương pháp Latin Hypercube Sampling
    # Ceil over a one-unit interval otherwise makes its lower integer unreachable.
    # Add feasible integer candidates so EI can consider both endpoints.
    integers = rng.integers(bounds[:,0].astype(int),bounds[:,1].astype(int)+1,size=(candidates,len(bounds))) #Tạo các ứng viên nguyên ngẫu nhiên trong phạm vi cho trước
    pool = np.vstack([pool,(integers-bounds[:,0])/(bounds[:,1]-bounds[:,0])]) #Chuẩn hóa các ứng viên nguyên và thêm chúng vào pool
    means,stds = model.predict(pool) #Dự đoán MAE và độ bất định cho các ứng viên
    ei = expected_improvement(means,stds,float(y.min())) #Tính Expected Improvement (EI) cho các ứng viên
    #Tối ưu EI từ nhiều điểm khởi tạo
    for start in pool[np.argsort(ei)[-12:]]:
        #Định nghĩa hàm âm EI
        def negative_ei(point):
            mu, sigma = model.predict(point)
            return -float(expected_improvement(mu,sigma,float(y.min()))[0])
        fit = minimize(negative_ei,start,method='L-BFGS-B',bounds=[(0,1)]*len(bounds)) #Tối ưu từ mỗi điểm khởi tạo
        pool = np.vstack([pool,fit.x]) #Thêm các điểm tối ưu vào pool
    decoded = np.unique(decode(pool,bounds),axis=0) #Chuyển ứng viên thành số nguyên và loại bỏ trùng lặp
    #Loại bỏ các bộ tham số đã thử
    tried = {tuple(row) for row in x}
    decoded = np.array([row for row in decoded if tuple(row) not in tried],dtype=int)
    if not len(decoded):
        raise RuntimeError('No unevaluated candidate left in acquisition pool')
    #Tính lại EI trên các ứng viên nguyên
    rounded_units = (decoded-bounds[:,0])/(bounds[:,1]-bounds[:,0])
    means,stds = model.predict(rounded_units)
    improvement = expected_improvement(means,stds,float(y.min()))
    #Chọn ứng viên có EI cao nhất
    chosen = int(np.argmax(improvement))
    #Lưu thông tin chẩn đoán
    diagnostics = dict(expected_improvement=float(improvement[chosen]), predicted_mae=float(means[chosen]),
        predicted_std=float(stds[chosen]), lengthscales=np.exp(model.theta[:-1]).tolist(),
        noise_variance_standardized=float(np.exp(model.theta[-1])), gp_fit_success=bool(model.result.success))
    return decoded[chosen], diagnostics

# Điều phối toàn bộ quá trình tối ưu
def minimize_objective(objective,bounds,initial_points=99,iterations=5,seed=13,on_trial=None,
                       *,resume_state=None,on_checkpoint=None):
    bounds = np.asarray(bounds,int)
    if bounds.shape!=(3,2) or np.any(bounds[:,0]<1) or np.any(bounds[:,1]<=bounds[:,0]):
        raise ValueError('Need three positive, nonconstant integer bounds')
    if initial_points<2 or iterations<0 or seed<0:
        raise ValueError('Need >=2 initial points, >=0 EI iterations and nonnegative seed')
    if initial_points+iterations>int(np.prod(bounds[:,1]-bounds[:,0]+1)): #Kiểm tra xem ngân sách có vượt quá không gian tìm kiếm rời rạc hay không
        raise ValueError('Budget exceeds discrete search space')
    rng = np.random.default_rng(seed) #khởi tạo rng
    initial = decode(qmc.LatinHypercube(d=3,rng=rng).random(initial_points),bounds) #Tạo các điểm ban đầu bằng phương pháp Latin Hypercube Sampling
    history=list(resume_state['history']) if resume_state else [] #Lưu trữ lịch sử các thử nghiệm trước đó nếu có
    if resume_state: rng.bit_generator.state=resume_state['rng_state'] #Khôi phục trạng thái RNG nếu có
    seen={tuple(row[name] for name in NAMES) for row in history} #Lưu trữ các bộ tham số đã thử
    if len(history)>initial_points+iterations or len(seen)!=len(history):#Kiểm tra tính hợp lệ của lịch sử khôi phục
        raise ValueError('Invalid resumed history')
    #Kiểm tra tính hợp lệ của từng thử nghiệm trong lịch sử khôi phục
    for i,row in enumerate(history):
        point=np.array([row[name] for name in NAMES])
        if row['trial']!=i or np.any(point<bounds[:,0]) or np.any(point>bounds[:,1]) or not np.isfinite(row['validation_mae']):
            raise ValueError('Invalid resumed trial')
    #Tiến hành tối ưu hóa bằng cách thử nghiệm các điểm ban đầu và sau đó sử dụng EI để đề xuất các điểm tiếp theo
    for i in range(len(history),initial_points+iterations):
        diagnostics={}
        #Nếu đang trong giai đoạn thử nghiệm các điểm ban đầu, chọn một điểm từ danh sách ban đầu. Nếu không, đề xuất một điểm mới bằng cách sử dụng EI dựa trên lịch sử các thử nghiệm trước đó.
        if i<initial_points:
            point = initial[i]
            while tuple(point) in seen: #Nếu điểm đã được thử nghiệm, chọn một điểm mới từ danh sách ban đầu
                point = rng.integers(bounds[:,0],bounds[:,1]+1)
            phase='lhs'
        else:
            point,diagnostics=propose(history,bounds,rng) #Đề xuất một điểm mới bằng cách sử dụng EI dựa trên lịch sử các thử nghiệm trước đó
            phase='ei'
        params=dict(zip(NAMES,map(int,point)))
        value,extra=objective(params,i) #Đánh giá điểm mới bằng cách sử dụng hàm mục tiêu
        if not np.isfinite(value):
            raise ValueError('Objective returned a nonfinite score')
        record=dict(trial=i,phase=phase,**params,validation_mae=float(value),**diagnostics,**extra) #Lưu trữ thông tin về thử nghiệm mới
        history.append(record);seen.add(tuple(point))#Lưu trữ thông tin về thử nghiệm mới
        if on_checkpoint: on_checkpoint(dict(history=history,rng_state=rng.bit_generator.state)) 
        if on_trial: on_trial(record,history)
    best=min(history,key=lambda r:r['validation_mae'])
    return {name:best[name] for name in NAMES},history
