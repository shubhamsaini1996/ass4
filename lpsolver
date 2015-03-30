import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def frange(a,b):
    temp=a
    while(temp<b):
        yield(temp)
        temp+=0.01
        
class LPsolver:

    def __init__(self):                                                     # Initialising instance variables
        self.objective=None
        self.constraints=None
        self.table=None
        self.result=None
    
    def solve(self,L,M,option):
        return (self.Simplex(L,M))

    def Simplex(self,L,M):                                                  # Simplex function takes two lists as arguments
            
        # List L contains co-efficients of variables in the objective
        # e.g.- If objective is (2x+5y+7z) then L=[2,5,7]
        # List M contains co-efficients of variables in the constraints and the last element is the constant
        # List M is a nested list
        # e.g. - If the constraints are x<=200 , y<=10 , z<=200 and x+2y+3z<=700 then --  
        # M=[[1,0,0,200],[0,1,0,10],[0,0,1,200],[1,2,3,700]], i.e.,
        # 1.x+0.y+0.z<=200 , 0.x+1.y+0.z<=10 , 0.x+0.y+1.z<=200 and 1.x+2.y+3.z<=700
        
        from numpy import array
        self.objective=array(L,float)
        self.constraints=array(M,float)
        no_of_variables=len(self.objective)                                 # variable to store total number of variables
        no_of_equations=len(self.constraints)                               # variable to store total number of equations(constraints)
        self.table=array([[0 for i in range(no_of_variables+no_of_equations+2)] for j in range(no_of_equations+1)],float)   # initialising the augmented matrix
        for i in range(no_of_equations):
            for j in range(no_of_variables):
                self.table[i][j]=self.constraints[i][j]                     # filling up the matrix
        for i in range(no_of_variables):
            self.table[-1][i]=(-1)*self.objective[i]                        # filling up the matrix
        for i in range(no_of_equations+1):
            self.table[i][no_of_variables+i]=1                              # filling up the matrix
        for i in range(no_of_equations):
            self.table[i][-1]=self.constraints[i][-1]                       # filling up the matrix
        self.table[-1][-1]=0                                                # matrix ready
        for i in range(no_of_equations+1):
            self.table[i][no_of_variables+i]=self.table[i][-1]/self.table[i][no_of_variables+i]
        flag=0
        if(min(self.table[-1])>=0):                                         # testing terminal condition
            flag=1
        while(flag==0):                                                     # loop
            pivot_col=self.table[-1].tolist().index(min(self.table[-1]))    # finding column of pivot
            pivot_row,temp,flag2=-1,0,0                                         
            for i in range(no_of_equations):                                                        
                if(self.table[i][pivot_col]>0 and (pivot_row==-1 or self.table[i][-1]/self.table[i][pivot_col]<temp)):   # finding row of pivot
                    pivot_row=i
                    temp=self.table[i][-1]/self.table[i][pivot_col]
                    flag2=1
            if(flag2==0):
                return ("Unbound")                                          # terminal condition 
            pivot=self.table[pivot_row][pivot_col]
            for i in range(no_of_equations+1):                              # loop to set all other elements of pivot's column to 0 by row operations
                if(i==pivot_row):
                    continue
                scale=self.table[i][pivot_col]/pivot
                self.table[i]=self.table[i]-(scale*self.table[pivot_row])
            if(min(self.table[-1])>=0):
                flag=1                                                      # terminal condition
            x_pos,y_pos,count_1,count_2=0,0,0,0                             # variables to store boundary points
            for i in range(no_of_equations+1):
                if(self.table[i][0]!=0):
                    count_1+=1
                    x_pos=self.table[i][-1]/self.table[i][0]
                if(self.table[i][1]!=0):
                    count_2+=1
                    y_pos=self.table[i][-1]/self.table[i][1]
            if(count_1>1):
                x_pos=0
            if(count_2>1):
                y_pos=0
            plt.plot(x_pos,y_pos,"o",color="c")                             # plotting boundary points
        self.result=""                                                      # string to store final answer
        for i in range(no_of_variables):                                    # building up the string
            count,temp,flag=0,0,1
            for j in range(no_of_equations+1):
                if(self.table[j][i]!=0):
                    count+=1
                    if(count>1):
                        flag=0
                        break
                    temp=self.table[j][-1]/self.table[j][i]
            if(flag==1):
                self.result+="x_"+str(i+1)+" = "+str(temp)+" ; "
            else:
                self.result+="x_"+str(i+1)+"=0 ; "
        self.result+="Maximum value of objective = "+str(self.table[-1][-1])
        self.plot()
        return (self.result)                                                # return result

    def plot(self):
        Color=["b","g","r","y","m"]                                         # list of colors to be used
        j=0                                                                 # counter variable
        plt.xlabel("x axis")                                            
        plt.ylabel("y axis")                                            
        Legend=[]                                                           # initialising legend
        x_max,y_max=0,0
        for i in self.constraints:                                          
            if(i[0]!=0):
                temp=i[2]/i[0]
                if(temp>x_max):
                    x_max=temp
            if(i[1]!=0):
                temp=i[2]/i[1]
                if(temp>y_max):
                    y_max=temp
        for i in self.constraints:                                          # plotting constraint lines
            if(i[0]!=0 and i[1]!=0):
                plt.plot([0,i[2]/i[0]],[i[2]/i[1],0],color=Color[j%5])
                x=list(frange(0,i[2]/i[0]))
                y1=list(map(lambda X:(i[2]-i[0]*X)/i[1],x))
                y2=list(map(lambda X:0,x))
                plt.fill_between(x,y1,y2,color=Color[j%5],alpha=0.7)        # filling color
            elif(i[0]!=0):
                plt.axvline(i[2],color=Color[j%5])
                x=list(frange(0,i[2]/i[0]))
                y1=list(map(lambda X:y_max,x))
                y2=list(map(lambda X:0,x))
                plt.fill_between(x,y1,y2,color=Color[j%5],alpha=0.7)        # filling color
            else:
                plt.axhline(i[2],color=Color[j%5])
                x=list(frange(0,x_max))
                y1=list(map(lambda X:i[2]/i[1],x))
                y2=list(map(lambda X:0,x))
                plt.fill_between(x,y1,y2,color=Color[j%5],alpha=0.7)        # filling color
            Legend.append(mpatches.Patch(color=Color[j%5],label="Constraint "+str(j+1)))    # updating legend
            j+=1
        Legend.append(mpatches.Patch(color="w",label="The innermost region is the feasible region"))
        Legend.append(mpatches.Patch(color="w",label="and its boundary points are the critical points"))
        plt.legend(handles=Legend)                                          # putting legend on graph
        plt.show()                                                          # displaying graph

lps=LPsolver()                                                              # object creation
solution=lps.solve([3,2],[[2,1,18],[2,3,42],[3,1,24]],option="Simplex")     # calling Simplex method
print(solution)

class LinearSystemSolver:

    def __init__(self):                                                     # initialising instance variables
        self.A=None
        self.b=None
        self.max_iterations=None
        self.accuracy=None
        self.initial_guess=None
        self.augmented=None
        self.result=None
    
    def solve(self,A,B,method,max_iterations=10000,accuracy=0.0001,initial_guess=None):
        
        # max_iterations, accuracy, and initial_guess are optional arguments
        
        if(method=="gauss"):
            return (self.Gauss(A,B))
        elif(method=="gauss-jordan"):
            return (self.GaussJordan(A,B))
        else:
            return (self.GaussSiedel(A,B,max_iterations,accuracy,initial_guess))

    def Gauss(self,A,B):                                                    # Gauss function takes two lists as arguments
                                                                            # A is the co-efficient matrix, B is matrix of constants

        # Suppose we have to solve the system --
        # 1.x+1.y+1.z=10
        # 0.x+1.y+0.z=2
        # 0.x+0.y+1.z=5
        # Then --
        # A=[[1,1,1],[0,1,0],[0,0,1]]
        # B=[10,2,5]
        
        from numpy import array
        self.A=array(A,float)
        self.b=array(B,float)
        n=len(self.A)                                                       # finding order of co-efficient matrix
        self.augmented=array([[0 for i in range(n+1)] for j in range(n)],float)          # initialising the augmented matrix
        for i in range(n):
            for j in range(n):
                self.augmented[i][j]=self.A[i][j]                           # filling up the entries of matrix A
        for i in range(n):
            self.augmented[i][n]=self.b[i]                                  # filling up the entries of matrix b
        for i in range(n-1):
            if(self.augmented[i][i]==0):                                    # checking if the pivot is 0
                flag=0  
                for j in range(i+1,n):
                    if(self.augmented[j][i]!=0):                            # swapping rows if pivot is 0
                        self.augmented[i],self.augmented[j]=self.augmented[j],self.augmented[i]          
                        flag=1
                        break
                if(flag==0):                                                
                    continue                                                # continue if the column contains only 0s
            for j in range(i+1,n):
                scale=self.augmented[j][i]/self.augmented[i][i]                                    
                self.augmented[j]=self.augmented[j]-scale*self.augmented[i] # performing row operations to make the entries below the pivot=0
        self.result=[0 for i in range(n)]                                   # initialisng the matrix containing the solution matrix
        for i in range(n-1,-1,-1):
            s=0
            for j in range(n):                                              # solving using back substitution
                s+=self.augmented[i][j]*self.result[j]
            self.result[i]=(self.augmented[i][n]-s)/self.augmented[i][i]
        return(self.result)                                                 # returning solution matrix

    def GaussJordan(self,A,B):                                              # GaussJordan function takes two lists as arguments
                                                                            # A is the co-efficient matrix, B is matrix of constants

        # Suppose we have to solve the system --
        # 1.x+1.y+1.z=6
        # 2.x+1.y+(-1).z=1
        # (-1).x+2.y+2.z=9
        # Then --
        # A=[[1,1,1],[2,1,-1],[-1,2,2]]
        # B=[6,1,9]
        
        from numpy import array
        self.A=array(A,float)
        self.b=array(B,float)
        n=len(self.A)                                                       # finding order of co-efficient matrix
        self.augmented=array([[0 for i in range(n+1)] for j in range(n)],float)          # initialising the augmented matrix
        for i in range(n):
            for j in range(n):
                self.augmented[i][j]=self.A[i][j]                           # filling up the entries of matrix A
        for i in range(n):
            self.augmented[i][n]=self.b[i]                                  # filling up the entries of matrix b
        for i in range(n):
            if(self.augmented[i][i]==0):                                    # checking if the pivot is 0
                flag=0
                for j in range(n):
                    if(self.augmented[j][i]!=0):                            # swapping rows if pivot is 0          
                        self.augmented[i],self.augmented[j]=self.augmented[j],self.augmented[i]
                        flag=1
                        break
                if(flag==0):
                    continue                                                # continue if the column contains only 0s
            for j in range(n):
                if(j==i):
                    continue
                scale=self.augmented[j][i]/self.augmented[i][i]
                self.augmented[j]=self.augmented[j]-scale*self.augmented[i] # performing row operations to make the entries in the the pivot column=0
        self.result=[0 for i in range(n)]                                   # initialisng the matrix containing the solution matrix
        for i in range(n):                                                  
            self.result[i]=self.augmented[i][n]/self.augmented[i][i]        # solving
        return(self.result)                                                 # returning solution matrix
    
    def GaussSiedel(self,A,B,max_iterations,accuracy,initial_guess):        # GaussSiedel function takes two lists 
                                                                            # (A is the co-efficient matrix, B is matrix of constants),         
                                                                            # maximmum allowed iterations, desired accuracy and initial guess as argument

        # Suppose we have to solve the system --
        # 1.x+1.y+1.z=10
        # 0.x+1.y+0.z=2
        # 0.x+0.y+1.z=5
        # Then --
        # A=[[1,1,1],[0,1,0],[0,0,1]]
        # B=[10,2,5]
        # max_iterations (optional argument) is the maximum number of iterations 
        # accuracy (optional argument) is the desired amount of Accuracy
        # initial_guess (optional argument) is the starting matrix, e.g.--
        #initial_guess=[1,1,1]
        
        from numpy import matrix     
        from numpy import ones_like                                       
        from numpy.linalg import inv
        self.A=A
        self.b=B
        self.max_iterations=max_iterations
        self.accuracy=accuracy
        if(initial_guess!=None):
            self.initial_guess=initial_guess
        else:
            self.initial_guess=ones_like(self.b)
        n=len(self.A)                                                       # finding order of co-efficient matrix
        mat=matrix(self.A)                                                  # defining co-efficient matrix
        self.b=matrix(self.b).transpose()                                   # defining constant matrix
        L=[[0 for i in range(n)] for j in range(n)]                         # initialising lower triangular matrix
        for i in range(n):
            for j in range(i,n):
                L[j][i]=self.A[j][i]                                        # filling up lower triangular matrix
        L=matrix(L)                                                                   
        U=mat-L                                                             # defining strict upper triangular matrix
        T=((-1)*inv(L))*U                                                   # calculating -U*(inverse of L)     
        C=inv(L)*self.b                                                     # calculating (inverse of L)*b
        diff,count=1,0                                                      # count variable counts number of iterations     
        self.result=matrix(self.initial_guess).transpose()                    # initialising solution matrix
        while(diff>accuracy and count<max_iterations):                      # iteration loop
            x_new=T*self.result+C                                           # updating x
            diff=(abs(x_new-self.result)).max()                             # finding accuracy
            self.result=x_new
            count+=1
        if(diff>accuracy):
            return("Iterations exceed limit")                               # maximum iterations exceeded 
        else:
            return(((self.result.transpose()).ravel()).tolist())            # return solution matrix as a list

lss=LinearSystemSolver()                                                    # object creation
for method in ["gauss","gauss-jordan","gauss-siedel"]:
    solution=lss.solve([[4,-1,-1],[-2,6,1],[-1,1,7]],[3,9,-6],method,10000,0.001,[0,0,0])
    print(solution)
