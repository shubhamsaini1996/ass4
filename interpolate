class Interpolate:
    
    def __init__(self):
        self.x_values=None
        self.fx_values=None
        self.result=None
        self.polynomial=None
        self.polylist=None

    def solve(self,L,M,method):
        if(method=="newton"):
            return (self.Newton(L,M))
        else:
            return (self.Lagrange(L,M))

    def Lagrange(self,L,M):                                                 # Lagrange function takes two lists as argument
        
        # L contains the list of x values
        # M contains the list of f(x) values
        # e.g.-
        #L=[1,2,3] , M=[0,-1,0]
        # i.e., f(1)=0, f(2)=-1, f(3)=0
        
        self.x_values=L
        self.fx_values=M
        self.polylist=[]
        n=len(self.x_values)                                                # n=length of L, i.e., the number of points
        w=(-1*self.x_values[0],1)                                           # initialising polynomial w
        for i in range(1,n):
            w=P.polymul(w,(-1*self.x_values[i],1))                          # calculating w
        self.result=array([0.0 for i in range(len(w)-1)])                        # initialising result array
        derivative=P.polyder(w)                                             # derivative of w
        for i in range(n):
            self.polylist.append((P.polydiv(w,(-1*self.x_values[i],1))[0]*self.fx_values[i])/P.polyval(self.x_values[i],derivative))# calculating result
            self.result+=self.polylist[-1]
        self.result=list(self.result)                                                 # list of co-efficients
        self.polynomial=""                                                  # string to store final polynomial
        for i in range(len(self.result)-1,0,-1):                                 # building up the string
            if(self.result[i]!=0):
                if(self.result[i]>0 and i!=(len(self.result)-1)):
                    self.polynomial+=" + "+str(self.result[i])+"x^"+str(i)+" "
                elif(self.result[i]>0 and i==(len(self.result)-1)):
                    self.polynomial+=str(self.result[i])+"x^"+str(i)+" "
                else:
                    self.polynomial+=" - "+str(-1*self.result[i])+"x^"+str(i)+" "
        if(self.result[0]!=0):
            self.polynomial+=" + "+str(self.result[0]) if self.result[0]>0 else " - "+str(-1*self.result[0])
        self.plot()
        return (self.polynomial)                                            # return result

    def plot(self):
        Color=["b","g","r","y","m"]                                         # list of colors to be used
        Legend=[]                                                           # initialising legend
        for i in range(len(self.polylist)):                                 # plotting polynomials
            x=list(frange(min(self.x_values)-1,max(self.x_values)+1))
            y=list(map(lambda num:P.polyval(num,self.polylist[i]),x))
            plt.plot(x,y,linewidth=2.0,color=Color[i%5])
            Legend.append(mpatches.Patch(color=Color[i%5],label="Polynomial "+str(i+1)))    # updating legend
        x=list(frange(min(self.x_values)-1,max(self.x_values)+1))
        y=list(map(lambda num:P.polyval(num,array(self.result)),x))
        plt.plot(x,y,linewidth=3.0,color="k")                               # plotting final polynomial
        Legend.append(mpatches.Patch(color="k",label="Final polynomial"))   
        x=self.x_values
        y=list(map(lambda num:P.polyval(num,array(self.result)),x))         
        plt.plot(x,y,"o",color="c")                                         
        plt.axis("equal")
        plt.axvline(0,color="k")
        plt.axhline(0,color="k")
        plt.xlabel(" x values ")
        plt.ylabel("f(x) values")
        plt.legend(handles=Legend)                                          # putting legend on the plot
        plt.show()                                                          # displaying plot

    def Newton(self,L,M):                                                   # Newton function takes two lists as arguments

        # L contains the list of x values
        # M contains the list of f(x) values
        # e.g.-
        #L=[1,2,3] , M=[0,-1,0]
        # i.e., f(1)=0, f(2)=-1, f(3)=0
        
        self.x_values=L
        self.fx_values=M
        n=len(self.x_values)                                                # n=length of L, i.e., the number of points
        mat=[[0.0 for i in range(n)] for j in range(n)]                     # initialising an n*n matrix 
        for i in range(n):                                                  # filling 1st column of matrix with f(x) values
            mat[i][0]=self.fx_values[i]
        for i in range(1,n):                                                # calculating entries of matrix
            for j in range(n-i):
                mat[j][i]=(mat[j+1][i-1]-mat[j][i-1])/(self.x_values[j+i]-self.x_values[j])
        # The matrix is of the form (for 4 points - x,y,z,w)
        #    f(x)    f(x,y)    f(x,y,z)    f(x,y,z,w)
        #    f(y)    f(y,z)    f(y,z,w)    0
        #    f(z)    f(z,w)    0           0 
        #    f(w)    0         0           0
        
        result=array((mat[0][0],))                                          # initialising result array
        for i in range(1,n):
            prod=(-1*self.x_values[0],1)                                    # initialising prod polynomial which is to be multiplied
                                                                            # with corresponding element of matrix mat
            for j in range(1,i):
                prod=P.polymul(prod,(-1*self.x_values[j],1))                # calculating prod    
            result=P.polyadd(result,array(prod)*mat[0][i])                  # calculating result
        result=list(result)                                                 # list of co-efficients
        self.polynomial=""                                                  # string to store final polynomial
        for i in range(len(result)-1,0,-1):                                 # building up the string
            if(result[i]!=0):
                if(result[i]>0 and i!=(len(result)-1)):
                    self.polynomial+=" + "+str(result[i])+"x^"+str(i)+" "
                elif(result[i]>0 and i==(len(result)-1)):
                    self.polynomial+=str(result[i])+"x^"+str(i)+" "
                else:
                    self.polynomial+=" - "+str(-1*result[i])+"x^"+str(i)+" "
        if(result[0]!=0):
            self.polynomial+=" + "+str(result[0]) if result[0]>0 else " - "+str(-1*result[0])
        return (self.polynomial)                                            # return result

apx=Interpolate()                                                          # object creation
for method in ["newton","lagrange"]:
    solution=apx.solve([-9,-4,-1,7],[5,2,-2,9],method)
    print(solution)
