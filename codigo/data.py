import numpy as np

def read_movies(filename_rating,filename_tags,sep_r,sep_t):
	movie_index = dict()
	set_movies_r = set()
	set_movies_t = set()
	set_movies = set()
	counter_index = 0
	reader = open(filename_rating)
	for line in reader:
		line = line.strip()
		data = line.split(sep_r)
		set_movies_r.add(data[1])
	reader.close()
	
	reader = open(filename_tags)
	for line in reader:
		line = line.strip()
		data = line.split(sep_t)
		set_movies_t.add(data[0])
	reader.close()
	
	print len(set_movies_r),"movies in ratings"
	print len(set_movies_t),"movies in tags"
	
	
	set_movies = set_movies_r & set_movies_t
	
	for m in set_movies:
		movie_index[m] = counter_index
		counter_index+=1
	
	print counter_index, "movies in both"
	
	return movie_index

def read_users(filename,sep):
	user_index = dict()
	set_users = set()
	counter_index = 0
	reader = open(filename)
	for line in reader:
		line = line.strip()
		data = line.split(sep)
		set_users.add(data[0])
	reader.close()
	
	for u in set_users:
		user_index[u] = counter_index
		counter_index+=1
	
	print counter_index,"users"	
	
	return user_index

def read_tags(filename,sep):
	tag_index = dict()
	set_tags = set()
	counter_index = 0
	reader = open(filename)
	for line in reader:
		line = line.strip()
		data = line.split(sep)
		set_tags.add(data[0])
	reader.close()

	for t in set_tags:
		tag_index[t] = counter_index
		counter_index+=1
	
	print counter_index,"tags"
	
	return tag_index



def read_movielens(filename):
	reader = open(filename)
	n = 0
	for line in reader:
		line = line.strip()
		print line
		n+=1
		if n==10:
			break
	
	reader.close()

def get_rating_matrix(filename_train,filename_test,index_u,index_i,sep_r):
	U = np.zeros((len(index_u),len(index_i)))
	J = np.zeros((len(index_u),len(index_i)))
	reader = open(filename_train)
	for line in reader:
		line = line.strip()
		data = line.split(sep_r)
		if data[0] in index_u and data[1] in index_i:
			U[index_u[data[0]]][index_i[data[1]]]=data[2]
			if data[2]>0:
				J[index_u[data[0]]][index_i[data[1]]]=1
	reader.close()
	reader = open(filename_test)
	for line in reader:
		line = line.strip()
		data = line.split(sep_r)
		if data[0] in index_u and data[1] in index_i:
			U[index_u[data[0]]][index_i[data[1]]]=0
			J[index_u[data[0]]][index_i[data[1]]]=0
	reader.close()

	return U,J
	
def get_relevance_matrix(filename_tags,index_i,index_t,sep_t):
	G = np.zeros((len(index_i),len(index_t)))
	
	reader = open(filename_tags)
	for line in reader:
		line = line.strip()
		data = line.split(sep_t)
		if data[0] in index_i and data[1] in index_t:
			G[index_i[data[0]]][index_t[data[1]]]=data[2]
	reader.close()
	return G



def save_preference_matrix(U,G,index_u,index_t,index_i,output):
	writer = open(output,'w')
	
	T = np.zeros((len(index_u),len(index_t)))
	total = len(index_u)*len(index_t)
	n = 0
	
	for user in index_u:
		for tag in index_t:
			n+=1
			nz1 = np.nonzero(U[index_u[user],:])
			nz2 = np.nonzero(G[:,index_t[tag]])
			nonzero_counter = len(set(nz1[0]) & set(nz2[0]))
			T[index_u[user]][index_t[tag]] += np.dot(U[index_u[user],:],G[:,index_t[tag]])
			if nonzero_counter == 0:
				print user
			if nonzero_counter > 0:
				T[index_u[user]][index_t[tag]] = T[index_u[user]][index_t[tag]]/nonzero_counter
			
			
			writer.write(user+'::'+tag+'::'+str(T[index_u[user]][index_t[tag]])+'\n')
		if n%(3*len(index_t)) == 0:
			print (100.0*n)/total, "%"
		
	writer.close()

def get_preference_matrix(filename_p,index_u,index_t,sep):
	T = np.zeros((len(index_u),len(index_t)))
	
	reader = open(filename_p)
	for line in reader:
		line = line.strip()
		data = line.split(sep)
		#if data[0] in index_u and data[1] in index_t:
		T[index_u[data[0]]][index_t[data[1]]]=data[2]
	reader.close()
	return T
	

def error_function(X,Y,Z):
	diff_rating = J*(U-np.dot(X,Y.transpose()))
	diff_tags = T-np.dot(X,Z.transpose())
	return 0.5*np.linalg.norm(diff_rating,'fro')**2+(0.5*alpha)*np.linalg.norm(diff_tags,'fro')**2+(0.5*beta)*(np.linalg.norm(X,'fro')**2+np.linalg.norm(Y,'fro')**2+np.linalg.norm(Z,'fro')**2)
	

def gradient_descent(seed_number,factors,J,U,T,max_iteration=10):
	writer = open("GDM.txt","w")
	
	#seed for random numbers
	np.random.seed(seed_number)
	#initialize X, Y, Z with random number in range (0,1)
	X = np.random.random((n_users,factors))
	Y = np.random.random((n_movies,factors))
	Z = np.random.random((n_tags,factors))
	writer.write("0 "+str(error_function(X,Y,Z))+"\n")
	print "error",error_function(X,Y,Z)
	
	
	it = 1
	
	#stop criteria
	while it <= max_iteration:
		print "iteration",it
		
		gradient_x = np.dot(J*(np.dot(X,Y.transpose())-U),Y)+alpha*np.dot((np.dot(X,Z.transpose())-T),Z)+beta*X
		gradient_y = np.dot((J*(np.dot(X,Y.transpose())-U)).transpose(),X)+beta*Y
		gradient_z = alpha*np.dot((np.dot(X,Z.transpose())-T).transpose(),X)+beta*Z
		
		gamma = 1.0
		print "gamma",gamma
		#select gamma that decreases error function value
		while error_function(X-gamma*gradient_x,Y-gamma*gradient_y,Z-gamma*gradient_z) > error_function(X,Y,Z):
			gamma = gamma/2
		
		print "gamma",gamma
		
		print "move matrix values in direction of the gradient"
		#move matrix values in direction of the gradient
		X = X-gamma*gradient_x
		Y = Y-gamma*gradient_y
		Z = Z-gamma*gradient_z
		
		print "error",error_function(X,Y,Z)
		writer.write(str(it)+" "+str(error_function(X,Y,Z))+"\n")
		
		it+=1
	writer.close()
	return np.dot(X,Y.transpose())

def write_predicted(input_file,output_file):
	reader = open(input_file)
	writer = open(output_file,'w')
	for line in reader:
		line = line.strip()
		data = line.split('::')
		user = data[0]
		item = data[1]
		rating = data[2] 
		
		if user in index_users and item in index_movies:
			predicted = str(A[index_users[user],index_movies[item]])
		else:
			predicted = '0'
		
		writer.write(user+"::"+item+"::"+rating+"::"+predicted+"\n")
		print user+"::"+item+"::"+rating+"::"+predicted
	reader.close()
	writer.close()



index_movies = read_movies('datasets/ml-1m/r1.train','datasets/tag-genome/tag_relevance.dat','::','\t')
index_users = read_users('datasets/ml-1m/users.dat','::')
index_tags = read_tags('datasets/tag-genome/tags.dat','\t')

#sizes
n_movies = len(index_movies)
n_users = len(index_users)
n_tags = len(index_tags)

print "get rating matrix"
#rating matrix
U,J = get_rating_matrix('datasets/ml-1m/r5.train','datasets/ml-1m/r5.test',index_users,index_movies,'::')

print "get relevance matrix"
#relevance matrix
G = get_relevance_matrix('datasets/tag-genome/tag_relevance.dat',index_movies,index_tags,'\t')

print U[index_users['270'],:]
print np.count_nonzero(U[index_users['270'],:])


raw_input()

exit()

print "get preference matrix"

factors = 3

X = np.random.random((n_users,factors))
Y = np.random.random((n_movies,factors))
Z = np.random.random((n_tags,factors))

alpha = 0
beta = 1

seed = 1234
max_iterations = 200

i = 1
print "i",i
#preference matrix
#save_preference_matrix(U,G,index_users,index_tags,index_movies,'preference'+str(i)+'.dat')
T = get_preference_matrix('preference'+str(i)+'.dat',index_users,index_tags,'::')



A = gradient_descent(seed,factors,J,U,T,max_iterations)

write_predicted('datasets/ml-1m/r1.test','predicted'+str(i)+'.txt')


