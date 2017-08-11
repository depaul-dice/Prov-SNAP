#!/usr/bin/python3

import re
from subprocess import call
import os, sys, time, datetime, glob, json
import argparse
import math
import time




parser = argparse.ArgumentParser(description='Process provenance log file.')
parser.add_argument('--nosub', action="store_true", default=False)
parser.add_argument('--nofilter', action="store_true", default=False)
parser.add_argument('--withfork', action="store_true", default=False)
parser.add_argument('-f', action="store", dest="fin_name", default="provenance.cde-root.1.log")
parser.add_argument('-d', action="store", dest="dir_name", default=".")
# parser.add_argument('--withgraph', action="store_true", default=False)

args = parser.parse_args()

showsub = not args.nosub
filter = not args.nofilter
withfork = args.withfork
dir = args.dir_name
logfile = args.fin_name
# withgraph = args.withgraph


def makePathNode(path):
	filename = os.path.basename(path).replace('"', '\\"')
	node = path.replace('\\', '\\\\').replace('"', '\\"')
	return (node, filename)
def isFilteredPath(path):
	if re.match('(^\/dev($|\/))|(^\/proc($|\/))|(^\/sys($|\/))|(^\/tmp($|\/))|(^\/var\/((cache)|(lock)|(log)|(run)|(tmp))\/)', path):
		return False
	else:
		return True



allNodes = {}
class Node:
	def __init__(self, name, type, containedIn):
		self.name = name
		self.toNodes = {}               #nodes started after this
		self.fromNodes = {}             #nodes ended before this
		self.contains = {}
		self.containedIn = containedIn
		
		self.type = type                #Group=0 Process=1 File=2 
		self.type2 = type               #0 = None, 1 = Only Processes, 2 = Only Files, 3 = Mixed

		self.tfnum = 0
		self.tfnamecalculated = False
		if containedIn != None:
			containedIn.contains[name] = self
		self.tfaddname = ''

		self.toGroup = {}
		self.fromGroup = {}
		
		self.scc=None #stronglyConnectedComponent
		self.scc_index=0
		self.scc_lowest=0
		self.scc_on_stack=0

		self.nextVer=None

		self.mark = 0
		self.mark2 = 0

		if type!=0:
			allNodes[name] = self



	def print_(self):
		print(self.name, self.type2, id(self))
		if(len(self.toNodes)>0):
			print("\tTo:")
			for n in self.toNodes:
				print("\t\t", n, id(self.toNodes[n]))
		if(len(self.fromNodes)>0):
			print("\tFrom:")
			for n in self.fromNodes:
				print("\t\t", n, id(self.fromNodes[n]))

		#if self.containedIn.name == 'FullGraph':
		if(len(self.toGroup)>0):
			print("\tToGroup:")
			for n in self.toGroup:
				print("\t\t", n, id(self.toGroup[n]))
		if(len(self.fromGroup)>0):
			print("\tFromGroup:")
			for n in self.fromGroup:
				print("\t\t", n, id(self.fromGroup[n]))
		if(len(self.contains)>0):
			print("\tContains:")
			for n in self.contains:
				print("\t\t", n, id(self.contains[n]))

	def gettfname(self):
		if not self.tfnamecalculated:
			self.tfnamecalculated = True
			if self.containedIn == None:
				self.tfname = re.sub(r'[^0-9a-zA-Z_.]','_',self.tfaddname + self.name)
			else:
				if self.containedIn.gettfname() == 'FullGraph':
					self.tfname = re.sub(r'[^0-9a-zA-Z_.]','_',self.tfaddname + self.name)
				else:
					self.tfname = self.containedIn.gettfname() + '/' + re.sub(r'[^0-9a-zA-Z_.]','_',self.tfaddname + self.name)
			if self.name[:2]=='P#':
				self.tfname = self.tfname + '/Process'
		return self.tfname


def joinNodes(pn,cn,type=1):
	pn.toNodes[cn.name] = cn
	cn.fromNodes[pn.name] = pn
	pn.toGroup[cn.name] = cn
	cn.fromGroup[pn.name] = pn
	if type == 2:
		pn.fromNodes[cn.name] = cn
		cn.toNodes[pn.name] = pn
		pn.fromGroup[cn.name] = cn
		cn.toGroup[pn.name] = pn
		joinNodes.counter[1]+=1
	joinNodes.counter[0]+=1
joinNodes.counter=[0,0]

def containNode(pn,cn):
	if pn.containedIn == cn.containedIn:
		pn.contains[cn.name] = cn
		if cn.containedIn != None:
			_ = cn.containedIn.contains.pop(cn.name)
		cn.containedIn = pn

		for n in cn.toGroup:
			node = cn.toGroup[n]
			_ = node.fromGroup.pop(cn.name)
			if node!=pn:
				node.fromGroup[pn.name] = pn
				pn.toGroup[n] = node
		for n in cn.fromGroup:
			node = cn.fromGroup[n]
			_ = node.toGroup.pop(cn.name)
			if node!=pn:
				node.toGroup[pn.name] = pn
				pn.fromGroup[n] = node

		pn.type2 = pn.type2|cn.type2
	else:
		print('Cannot contain:',pn.name,'is in',pn.containedIn.name, 'but', cn.name, 'is in', cn.containedIn)

class SCC:
	allsccs = []
	num = 0
	def __init__(self):
		self.num = SCC.num
		SCC.num += 1
		SCC.allsccs.append(self)

		self.nodes = {}

	def addnode(self, n):
		self.nodes[n.name] = n
		self.recursenodesscc(n)
		#print(SCC.num)
	def recursenodesscc(self, node):
		node.scc = self
		for n2 in node.contains:
			self.recursenodesscc(node.contains[n2])



def findSCCs(allNodes):
	num=1
	for n in allNodes:
		if allNodes[n].scc_index==0:
			num = findscc(allNodes[n],num,[])
def findscc(node,num,s):
	node.scc_index=num
	node.scc_lowest=num
	s.append(node)
	node.scc_on_stack=True
	num+=1

	for n2 in node.toGroup:
		node2=node.toGroup[n2]
		if node2.scc_index==0:
			num = findscc(node2,num,s)
			if node2.scc_lowest<node.scc_lowest:
				node.scc_lowest = node2.scc_lowest
		elif node2.scc_on_stack and node2.scc_index<node.scc_lowest:
			node.scc_lowest = node2.scc_index

	if node.scc_lowest == node.scc_index:
		scc = SCC()
		while True:
			node2 = s.pop()
			node2.scc_on_stack = False
			scc.addnode(node2)
			if node2==node:
				break
	return num

		#print('TF Name = ', cn.tfname)
def comp1(node):
	l = list(node.toGroup.values())
	ids = [ id(x) for x in l ]    
	ids.sort()
	return ids
def comp2(node):
	l = list(node.fromGroup.values())
	ids = [ id(x) for x in l ]    
	ids.sort()
	return ids


def groupSimilar(topNode, num):

	templist = list(topNode.contains.values())
	templist.sort(key = lambda node: 1 if node.type2==2 else 0)
	templist.sort(key = lambda node: len(node.fromGroup))
	templist.sort(key = lambda node: len(node.toGroup))
	simlist = None
	i = 0
	while i < len(templist):
		onlyfiles = (templist[i].type2==2)
		len_fg = len(templist[i].fromGroup)
		len_tg = len(templist[i].toGroup)
		J = i + 1
		while J < len(templist):
			if (onlyfiles != (templist[J].type2==2)) or len_fg != len(templist[J].fromGroup) or len_tg != len(templist[J].toGroup):
				break
			J += 1
		if J-i > 1:
			templist[i:J] = sorted(templist[i:J], key=comp1)
			templist[i:J] = sorted(templist[i:J], key=comp2)

			while i < J:
				l_t = comp1(templist[i])
				l_f = comp2(templist[i])
				j = i + 1
				while j < J:
					if l_t != comp1(templist[j]):
						break
					if l_f != comp2(templist[j]):
						break
					j += 1
				if j-i > 1:
					simlist = (i,j,simlist)
					
				i = j
		else:
			i += 1

	while simlist!=None:
		i,j,simlist = simlist
		newnode = Node('Sim#' + str(num),0,topNode)
		while i < j:
			containNode(newnode,templist[i])
			i += 1
		l_0 = list(newnode.contains.items())
		for n,node in l_0:
			if n[:4]=='Sim#':
				for n2 in node.contains:
					node2=node.contains[n2]
					node2.containedIn = newnode
					newnode.contains[n2]=node2
				_ = newnode.contains.pop(n)
		num +=1

	return num

def groupSimilarFromProcess(topNode, num):

	templist = list(topNode.contains.values())
	templist.sort(key = lambda node: 1 if node.type2==2 else 0)
	templist.sort(key = lambda node: len(node.fromGroup))
	simlist = None
	i = 0
	while i < len(templist):
		onlyfiles = (templist[i].type2==2)
		len_fg = len(templist[i].fromGroup)
		if not onlyfiles:# or len_fg==0:
			i+=1
			continue
		J = i + 1
		while J < len(templist):
			if (onlyfiles != (templist[J].type2==2)) or len_fg != len(templist[J].fromGroup):
				break
			J += 1
		if J-i > 1:
			templist[i:J] = sorted(templist[i:J], key=comp2)

			while i < J:
				l_f = comp2(templist[i])
				j = i + 1
				while j < J:
					if l_f != comp2(templist[j]):
						break
					j += 1
				if j-i > 1:
					simlist = (i,j,simlist)
					
				i = j
		else:
			i += 1

	while simlist!=None:
		i,j,simlist = simlist
		newnode = Node('FromSim#' + str(num),0,topNode)
		while i < j:
			containNode(newnode,templist[i])
			i += 1
		l_0 = list(newnode.contains.items())
		for n,node in l_0:
			if n[:4]=='FromSim#':
				for n2 in node.contains:
					node2=node.contains[n2]
					node2.containedIn = newnode
					newnode.contains[n2]=node2
				_ = newnode.contains.pop(n)
		num +=1

	return num
def writeNode(fout, node, tfnum): 
	# Assumes files are already done
	if node.tfnum == 0:
		#print(node.name, node.scc, len(node.fromNodes), len(node.toNodes))
		l = []
		for n in node.toNodes:
			node2 = node.toNodes[n]
			#if node.type != 2 or node.scc!=node2.scc:
			if node2.tfnum == 0:
				tfnum = writeNode(fout, node2, tfnum) 
			l.append('_' + str(node2.tfnum))
		#print('len=',len(l))
		'''
		for n in node.toNodes:
			node2 = node.toNodes[n]
			if node2.type == 2 and node2.scc==node.scc: 
				if node2.tfnum == 0:
					tfnum = writeNode(fout, node2, tfnum)
				l.append('_' + str(node2.tfnum))
		'''
		node.tfnum = tfnum
		tfnum += 1
		if len(l)>1:
			fout.write('\t' + '_' + str(node.tfnum) + ' = tf.concat([')
			for v in l:
				fout.write( v + ',')
			fout.write('], axis = 0, name=\'' + node.gettfname() + '\')#' + str(id(node)) + '\n')
		elif len(l)==1:
			fout.write('\t' + '_' + str(node.tfnum) + ' = tf.identity(' + l[0] + ',name = \'' + node.gettfname() + '\')#' + str(id(node)) + '\n' )
		else:
			#fout.write('\t' + '_' + str(node.tfnum) + ' = tf.Variable([1], name=\'' + node.tfname + '\')\n' )
			fout.write('\t' + '_' + str(node.tfnum) + ' = tf.placeholder(tf.int32,shape=[1], name=\'' + node.gettfname() + '\')#' + str(id(node)) + '\n')
	return tfnum
def writeOutput(allNodes, fileNodes):
	if not os.path.exists(dir):
		os.makedirs(dir)
	os.system("rm -f " + dir + "/*.svg " + dir + "/*.gv " + dir + "/*.html")

	fout = open(dir + '/maintf.py', 'w')
	fout.write('#This is a script to generate the TensorFlow graph \n' +
			   '\nfrom __future__ import absolute_import \n' +
			   'from __future__ import division \n' +
			   'from __future__ import print_function \n' +
			   '\nimport argparse \n' +
			   'import sys \n' +
			   '\nimport tensorflow as tf \n' +
			   '\nFLAGS = None \n' +
			   '\ndef graphGenerator():\n' +
			   '\tsess = tf.InteractiveSession()\n')
	############
	tfnum = 1
	'''
	for n in fileNodes:
		node = fileNodes[n]
		if node.tfnum==0 and all(node.scc==node2.scc for node2 in list(node.fromNodes.values())):
			node.tfnum = tfnum
			tfnum += 1
			fout.write('\t' + '_' + str(node.tfnum) + ' = tf.placeholder(tf.int32,shape=[1], name=\'' + node.gettfname() + '\')\n')
	'''
	for n in allNodes:
		tfnum = writeNode(fout,allNodes[n],tfnum)



	############
	fout.write('\n\tgraphGenerator_writer = tf.summary.FileWriter(FLAGS.log_dir + \'/train\', sess.graph)\n' +
			   '\ttf.global_variables_initializer().run()\n' +
			   '\tgraphGenerator_writer.close()\n')
	fout.write('\ndef main(_):\n' +
			   '\tif tf.gfile.Exists(FLAGS.log_dir):\n' +
			   '\t\ttf.gfile.DeleteRecursively(FLAGS.log_dir)\n' +
			   '\ttf.gfile.MakeDirs(FLAGS.log_dir)\n' +
			   '\tgraphGenerator()\n'
			   )

	fout.write('\nif __name__ == \'__main__\':\n' +
			   '\tparser = argparse.ArgumentParser()\n' +
			   '\tparser.add_argument(\'--data_dir\', type=str, default=\'/tmp/ptu/input_data\', help=\'Directory for storing input data\')\n' +
			   '\tparser.add_argument(\'--log_dir\', type=str, default=\'/tmp/ptu/logs\', help=\'Summaries log directory\')\n' +
			   '\tFLAGS, unparsed = parser.parse_known_args()\n' +
			   '\ttf.app.run(main=main, argv=[sys.argv[0]] + unparsed)'
			   )

	fout.close()

	

def ungroup(node): #Note: does not update from/toGroup. Intended for use at end.
	if len(node.contains)==0:
		return
	elif len(node.contains)<3:
		while True:
			#print(len(node.contains),node.name)
			minc,minname,minnode = min((len(node2.contains) if node2.type == 0 else float('inf'),n2,node2) for n2,node2 in node.contains.items())
			#print(minc,minname,len(node.contains[minname].contains))
			if minc!=None and minc+len(node.contains)<10:
				for n2,node2 in minnode.contains.items():
					node2.containedIn = node
					node.contains[n2] = node2
				node.contains.pop(minname)
			else:
				break
	for n2, node2 in node.contains.items():
		ungroup(node2)

def divideGroupsArbitrarily(node): #Note: does not update from/toGroup. Intended for use at end.
	#print(len(node.contains))
	if len(node.contains)>9:

		size = int(min([math.sqrt(len(node.contains)),9]))
		l1 = list(node.contains.items())
		node.contains = {}
		for i in range(size):
			node.contains[node.name + '_G#' + str(i)] = Node(node.name + '_G#' + str(i),0,node)
		i=0
		for n2, node2 in l1:
			node2.containedIn = node.contains[node.name + '_G#' + str(i)]
			node2.containedIn.contains[n2] = node2
			i=(i+1)%size
	for n2, node2 in node.contains.items():
		divideGroupsArbitrarily(node2)
def divideSimGroupsArbitrarily(node): #Note: does not update from/toGroup. Intended for use at end.
	#print(len(node.contains))
	if len(node.contains)>9 and (node.name[:3]=='Sim' or node.name=='FromSim'):

		size = int(min([math.sqrt(len(node.contains)),9]))
		l1 = list(node.contains.items())
		node.contains = {}
		for i in range(size):
			node.contains[node.name + '_G#' + str(i)] = Node(node.name + '_G#' + str(i),0,node)
		i=0
		for n2, node2 in l1:
			node2.containedIn = node.contains[node.name + '_G#' + str(i)]
			node2.containedIn.contains[n2] = node2
			i=(i+1)%size
	for n2, node2 in node.contains.items():
		divideSimGroupsArbitrarily(node2)


def removeRedundentEdges(allNodes):
	i=1
	count = 0
	for node in list(allNodes.values()):
		for node2 in list(node.toNodes.values()):
			if node2.mark2 == i:
				node2.mark = -2
			else:
				node2.mark2 = i
				node2.mark = -1
				removeRedundentEdges_(node2, i)
		for node2 in list(node.toNodes.values()):
			if node2.mark==-2:
				node2.fromNodes.pop(node.name)
				node.toNodes.pop(node2.name)
				node2.fromGroup.pop(node.name)
				node.toGroup.pop(node2.name)
				count+=1
		i+=1
	print('Edges Removed =', count)

def removeRedundentEdges_(node, i):
	for node2 in list(node.toNodes.values()):
		if node2.mark2 == i:
			if node2.mark < 0:
				node2.mark = -2
			#else node2 has been visited
		else:
			node2.mark2 = i
			node2.mark = 1
			removeRedundentEdges_(node2, i)

def groupARCompatibleN(topNode,N):					#N = minimum number of nodes to make group(otherwise ungroup at end)
	print('----------------------------')
	groups = None									#Format [ 0: number, 1: contains, 2: markcount, 3: marklist, 4: isinlist, 5: next ]
	m = 0
	for node in topNode.contains.values():
		node.arcgroupnum = node.level if hasattr(node, 'level') else findlevel(node)
		m = max(m,node.arcgroupnum)
		node.arcmark = 0


	allGroups = []
	for i in range(m+1):
		groups = [i, set(), 0, None, True, groups]
		allGroups.append(groups)

	for node in topNode.contains.values():
		node.arcgroup= allGroups[node.arcgroupnum]
		node.arcgroup[1].add(node)

	nextnum = m+1
	while groups!=None:
		g = groups
		groups = groups[5]
		g[4] = False
		glist = None
		for node in g[1]:
			for node2 in node.fromGroup.values():
				if g[0]==19:
					print('Here')
				if node2.arcmark == 0:
					print('          -')
					node2.arcmark = 1
					node2.arcgroup[2]+=1
					node2.arcgroup[3]=[node2,node2.arcgroup[3]]
					if node2.arcgroup[2]==1:
						glist = [node2.arcgroup, glist]

		temp = glist
		print(g[0], len(g[1]), [node.name for node in g[1]], 'From')
		while temp!=None:
			print('\t',temp[0][0],len(temp[0][1]),temp[0][2])
			temp=temp[1]

		mark=0
		temp = glist
		while temp!=None:
			gr = temp[0]
			if gr[2]!=len(gr[1]):
				print('Break (From)', gr[0],nextnum)
				if gr == g:
					mark=1
				groups = [nextnum, set(), 0, None, True, groups]
				allGroups.append(groups)
				while gr[3]!=None:
					node = gr[3][0]
					node.arcgroupnum = nextnum
					node.arcmark = 0
					node.arcgroup = groups
					groups[1].add(node)
					print(node.name)
					gr[1].remove(node)
					gr[3]=gr[3][1]
				if not gr[4]:
					gr[5] = groups
					groups = gr
					gr[4] = True
				nextnum+=1
			else:
				while gr[3]!=None:
					gr[3][0].arcmark = 0
					gr[3] = gr[3][1]

			gr[2]=0
			temp=temp[1]

		if mark==0:
			glist = None
			for node in g[1]:
				for node2 in node.toGroup.values():
					if node2.arcmark == 0:
						node2.arcmark = 1
						node2.arcgroup[2]+=1
						node2.arcgroup[3]=[node2,node2.arcgroup[3]]
						if node2.arcgroup[2]==1:
							glist = [node2.arcgroup, glist]
						#print('\t\t\t', node2.arcgroup[0], len(node2.arcgroup[1]))

			temp = glist
			print(g[0], len(g[1]), 'To')
			while temp!=None:
				print('\t',temp[0][0],len(temp[0][1]),temp[0][2])
				temp=temp[1]

			temp = glist
			while temp!=None:
				gr = temp[0]
				if gr[2]!=len(gr[1]):
					print('Break (To)', gr[0],nextnum)
					groups = [nextnum, set(), 0, None, True, groups]
					allGroups.append(groups)
					while gr[3]!=None:
						node = gr[3][0]
						node.arcgroupnum = nextnum
						node.arcmark = 0
						node.arcgroup = groups
						groups[1].add(node)
						print(node.name)
						gr[1].remove(node)
						gr[3]=gr[3][1]
					if not gr[4]:
						gr[5] = groups
						groups = gr
						gr[4] = True
					nextnum+=1
				else:
					while gr[3]!=None:
						gr[3][0].arcmark = 0
						gr[3] = gr[3][1]
				gr[2]=0
				temp=temp[1]


	for g in allGroups:
		#print(g[0])
		if len(g[1])>=N:
			newnode = Node('ARCG#'+str(g[0]),0,topNode)
			for node in g[1]:
				containNode(newnode,node)
	print('----------------------------')


def groupARCompatibleNwithSameNumEdges(topNode,N):					#N = minimum number of nodes to make group(otherwise ungroup at end)
	print('----------------------------')
	class ARCGroup:
		def __init__(self, num, creationDivideNum):
			self.num = num
			self.vertexList = None               
			self.isinlist = False
			self.divideGroup = None
			self.divideNum = None
			self.creationDivideNum = creationDivideNum
			self.groupnode=None

	filegroup = ARCGroup(0,0)
	processgroup = ARCGroup(1,0)
	gnum = 2
	for node in topNode.contains.values():
		if node.type2==2:
			g = filegroup
		else:
			g = processgroup
		node.arcgroup = g
		node.arcgroupnext = g.vertexList
		node.arcgroupprev = None
		if g.vertexList!=None:
			g.vertexList.arcgroupprev = node
		g.vertexList = node
	glist = (filegroup,(processgroup,None))
	filegroup.isinlist=True
	processgroup.isinlist=True
	
	divideNum = 0
	while glist!=None:
		g = glist[0]
		glist = glist[1]
		g.isinlist = False
		gvlist = None
		v = g.vertexList
		while v!=None:
			gvlist = (v,gvlist)
			v = v.arcgroupnext
		
		
		#divide using 'fromGroup' of g
		
		divideNum+=1
		dividelist=None
		
		gv = gvlist
		while gv!=None:
			v = gv[0]
			gv = gv[1]
			for v2 in v.fromGroup.values():

				g2 = v2.arcgroup
				#Make new group if not exists
				if g2.divideNum!=divideNum:
					if g2.creationDivideNum!=divideNum:
						dividelist = (g2,dividelist)
					g2.divideGroup = ARCGroup(gnum,divideNum)
					g2.divideNum = divideNum
					gnum+=1
				g3 = g2.divideGroup
				#Remove from Previous Group
				if v2.arcgroupprev==None:
					g2.vertexList = v2.arcgroupnext
				else:
					v2.arcgroupprev.arcgroupnext = v2.arcgroupnext
				if v2.arcgroupnext!=None:
					v2.arcgroupnext.arcgroupprev = v2.arcgroupprev
				#Add to new group
				v2.arcgroup = g3
				if g3.vertexList!=None:
					g3.vertexList.arcgroupprev = v2
				v2.arcgroupnext = g3.vertexList
				v2.arcgroupprev = None
				g3.vertexList = v2
		#Add groups to glist
		mark=0
		while dividelist!=None:
			g2 = dividelist[0]
			dividelist=dividelist[1]

			g3 = g2
			while g3.vertexList==None:#g3 cannot become None
				g3=g3.divideGroup

			if g3.divideNum==divideNum:
				if g==g2:
					mark=1
				if not g3.isinlist:
					g3.isinlist=True
					glist = (g3,glist)
				g3 = g3.divideGroup
				while g3!=None:
					if g3.vertexList!=None:
						g3.isinlist=True
						glist = (g3,glist)
					g3 = g3.divideGroup
			elif g2.isinlist:
				g3.isinlist=True
				glist = (g3,glist)
		
		#Check if g got divided
		if mark==1:
			continue
		
		#divide using 'toGroup' of g
		
		divideNum+=1
		dividelist=None
		
		gv = gvlist
		while gv!=None:
			v = gv[0]
			gv = gv[1]
			for v2 in v.toGroup.values():
				g2 = v2.arcgroup
				#Make new group if not exists
				if g2.divideNum!=divideNum:
					if g2.creationDivideNum!=divideNum:
						dividelist = (g2,dividelist)
					g2.divideGroup = ARCGroup(gnum,divideNum)
					g2.divideNum = divideNum
					gnum+=1
				g3 = g2.divideGroup
				#Remove from Previous Group
				if v2.arcgroupprev==None:
					g2.vertexList = v2.arcgroupnext
				else:
					v2.arcgroupprev.arcgroupnext = v2.arcgroupnext
				if v2.arcgroupnext!=None:
					v2.arcgroupnext.arcgroupprev = v2.arcgroupprev
				#Add to new group
				v2.arcgroup = g3
				if g3.vertexList!=None:
					g3.vertexList.arcgroupprev = v2
				v2.arcgroupnext = g3.vertexList
				v2.arcgroupprev = None
				g3.vertexList = v2
		#Add groups to glist
		while dividelist!=None:
			g2 = dividelist[0]
			dividelist=dividelist[1]

			g3 = g2
			while g3.vertexList==None:#g3 cannot become None
				g3=g3.divideGroup

			if g3.divideNum==divideNum:
				if not g3.isinlist:
					g3.isinlist=True
					glist = (g3,glist)
				g3 = g3.divideGroup
				while g3!=None:
					if g3.vertexList!=None:
						g3.isinlist=True
						glist = (g3,glist)
					g3 = g3.divideGroup
			elif g2.isinlist:
				g3.isinlist=True
				glist = (g3,glist)

	for node in list(topNode.contains.values()):
		if node.arcgroupprev!=None or node.arcgroupnext!=None:
			g = node.arcgroup
			if g.groupnode==None:
				g.groupnode = Node('ARCG#'+str(g.num),0,topNode)
			containNode(g.groupnode,node)
	

def findlevel(node):
	m = 0
	for node2 in node.toGroup.values():
		m = max(m, 1+(node2.level if hasattr(node2, 'level') else findlevel(node2)))
	node.level = m
	return m

def main():
	# open input output files
	try:
		fin = open(logfile, 'r')
	except IOError:
		print("Error: can\'t find file " + logfile + " or read data\n")
		sys.exit(-1)

	while 1:  
		line = fin.readline()
		if re.match('^#.*$', line) or re.match('^$', line):
			continue
		else:
			break


	topNode = Node("FullGraph", 0, None)
	
	processNodes = {}
	fileNodes = {}
	tfaddname = {}
	
	spawnedProcesses = {} #P_Name: Actual P_Name
	processes = {}        #P_Name:  ( readFiles(set of names), writeFiles(set of names) )
	files = {}            #Name: (version, isRead)

	while 1: 
		if re.match('^#.*$', line) or re.match('^$', line):
			if re.match('^# @.*$', line):
				line = fin.readline()
				continue
		line = line.rstrip('\n').replace('\\', '\\\\').replace('"', '\\"')
		words = line.split(' ', 6)
		pid = 'P#' + words[1]
		action = words[2]
		path = '' if len(words) < 4 else words[3]
		path = path.replace('"', '\"')
		
		if action!='EXIT':
			while pid in spawnedProcesses:
				pid = spawnedProcesses[pid]
		if pid in processes:
			p1 = processes[pid]
		else:
			p1 = (set(),set())
			processes[pid]=p1

		if action == 'SPAWN':
			pid2 = 'P#' + words[3]
			spawnedProcesses[pid2] = pid


		elif action == 'EXECVE':
			pid2 = 'P#' + words[3]
			if pid2 in spawnedProcesses:
				spawnedProcesses.pop(pid2)
			
			if pid2 in processes:
				p2 = processes[pid2] 
			else:
				p2 = (set(),set())
				processes[pid2] = p2
			
			for f in p1[0]:
				p2[0].add(f)

			
			path = '' if len(words) < 5 else words[4]
			path = path.replace('"', '\"')
			(pnode, filename) = makePathNode(path)
			pnode = filename + ' [' + pnode + ']'

			tfaddname[pid2] = filename+':'
			if isFilteredPath(path):
				p2[0].add(pnode)


		elif action == 'READ' and isFilteredPath(path):  
			(pnode, filename) = makePathNode(path)
			pnode = filename + ' [' + pnode + ']'
			p1[0].add(pnode)

		elif action == 'WRITE' and isFilteredPath(path):  
			(pnode, filename) = makePathNode(path)
			pnode = filename + ' [' + pnode + ']'
			p1[1].add(pnode)

		elif action == 'READ-WRITE' and isFilteredPath(path):  
			(pnode, filename) = makePathNode(path)
			pnode = filename + ' [' + pnode + ']'
			p1[0].add(pnode)
			p1[1].add(pnode)

		elif action == 'EXIT':
			if len(p1[1])>0:
				node = Node(pid,1,topNode)
				processNodes[pid] = node
				node.tfaddname = tfaddname[pid]
				for f in p1[0]:
					if f in files:
						(ver,_) = files[f]
						files[f] = (ver, True)
						fname = f if ver==1 else f + '_V' + str(ver)
					else:
						files[f] = (1, True)
						fname = f
					if fname not in fileNodes:
						fnode = Node(fname,2,topNode)
						fileNodes[fname] = fnode
					else:
						fnode = fileNodes[fname]
					joinNodes(fnode,node)
				for f in p1[1]:
					prevVer = None
					if f in files:
						(ver,isRead) = files[f]
						if isRead==True:
							prevVer = f if ver==1 else f + '_V' + str(ver)
							ver+=1
							files[f] = (ver, False)
						fname = f if ver==1 else f + '_V' + str(ver)
					else:
						files[f] = (1, False)
						fname = f
					if fname not in fileNodes:
						fnode = Node(fname,2,topNode)
						if prevVer!=None:
							fileNodes[prevVer].nextVer = fnode
						fileNodes[fname] = fnode
					else:
						fnode = fileNodes[fname]
					joinNodes(node,fnode)
		line = fin.readline()
		if line == '':
			break

	fin.close()
	'''
	for n,node in fileNodes.items():
		print(n)
	
	#Debug
	for n in allNodes:
		allNodes[n].print_()
	
	print('------')
	for n in allNodes:
		print(n)
	'''
	print("Vertices :", len(allNodes), "Edges :", joinNodes.counter)
	########## Graph Summarization
	'''
	#Remove unconnected nodes
	print('Vertices =', len(topNode.contains), 'Edges =', sum([len(topNode.contains[n].fromGroup) for n in topNode.contains]))
	toRemove = set()
	for n in allNodes:
		node = allNodes[n]
		if len(node.fromNodes)==0 and len(node.toNodes)==0:
			toRemove.add(n)
	for n in toRemove:
		_ = topNode.contains.pop(n)
		_ = allNodes.pop(n)
	print("Node Remove Count =", len(toRemove))
	print('Vertices =', len(topNode.contains), 'Edges =', sum([len(topNode.contains[n].fromGroup) for n in topNode.contains]))
	

	#Group Strongly connected components
	print('Vertices =', len(topNode.contains), 'Edges =', sum([len(topNode.contains[n].fromGroup) for n in topNode.contains]))
	findSCCs(topNode.contains)
	count=0
	for scc in SCC.allsccs:
		if len(scc.nodes)>1:
			newnode = Node('SCC#'+str(count),0,topNode)
			count+=1
			for n in scc.nodes:
				node=scc.nodes[n]
				containNode(newnode,node)

	print('Num of SCCs =', len(SCC.allsccs), 'Largest=', max([len(scc.nodes) for scc in SCC.allsccs]), 'Group Count =', count)
	print('Vertices =', len(topNode.contains), 'Edges =', sum([len(topNode.contains[n].fromGroup) for n in topNode.contains]))
	'''

	#Remove links which have equivalent longer paths of legnth 2 or more
	#removeRedundentEdges(allNodes)


	num=0
	#Group some processes with rw
	for f in files:
		node = fileNodes[f]
		while node.nextVer!=None:
			if len(node.fromGroup)==1 and len(node.toGroup)==1:
				if list(node.toGroup.keys())[0] in node.nextVer.fromGroup:
					n2, node2 = list(node.toGroup.items())[0]
					ok = True
					for n3 in node2.fromGroup:
						node3 = node2.fromGroup[n3]
						if node3 != node and len(node3.fromGroup)>0:
							ok = False
							break
					#print(node.name, ok)
					if ok:
						newnode = Node('P_#' + str(num),0,topNode)
						num+=1
						containNode(newnode, list(node.fromGroup.values())[0])
						containNode(newnode, node)
						containNode(newnode, node2)
						#print('ok',num)
			node=node.nextVer
	print('num=',num)

	'''
	for n in allNodes:
		node = allNodes[n]
		for n2 in node.fromNodes:
			node2 = node.fromNodes[n2]
			if n not in node2.toNodes:
				print(n,n2)
		for n2 in node.toNodes:
			node2 = node.toNodes[n2]
			if n not in node2.fromNodes:
				print(n,n2)
	for n in allNodes:
		node = allNodes[n]
		for n2 in node.fromGroup:
			node2 = node.fromGroup[n2]
			if n not in node2.toGroup:
				print(n,n2)
		for n2 in node.toGroup:
			node2 = node.toGroup[n2]
			if n not in node2.fromGroup:
				print(n,n2)
	'''
	'''
	#Grouping files with exactly one creator process or exactly one process using it  with the process
	for fn in fileNodes:
		node = fileNodes[fn]
		if len(node.fromNodes) == 1:
			for n in node.fromNodes:
				node2 = node.fromNodes[n]
			if node2.containedIn == topNode:
				newP = Node(node2.name+'#',0,topNode)
				containNode(newP, node2)
			else:
				newP = node2.containedIn
			containNode(newP, node)
		
		elif len(node.toNodes) == 1 and len(node.fromNodes) == 0:
			for n in node.toNodes:
				node2 = node.toNodes[n]
			if node2.containedIn == topNode:
				newP = Node(node2.name+'#',0,topNode)
				containNode(newP, node2)
			else:
				newP = node2.containedIn
			containNode(newP, node)
	print('Vertices =', len(topNode.contains), 'Edges =', sum([len(topNode.contains[n].fromGroup) for n in topNode.contains]))
	'''

	lastnum = -1
	#num = groupSimilarFromProcess(topNode, num)
	#print(num)
	groupARCompatibleNwithSameNumEdges(topNode,2)
	print('Vertices =', len(topNode.contains), 'Edges =', sum([len(topNode.contains[n].fromGroup) for n in topNode.contains]))
	groupARCompatibleNwithSameNumEdges(topNode,2)
	print('Vertices =', len(topNode.contains), 'Edges =', sum([len(topNode.contains[n].fromGroup) for n in topNode.contains]))
	
	num = groupSimilar(topNode, num)
	print('S\t', num, 'Vertices =', len(topNode.contains), 'Edges =', sum([len(topNode.contains[n].fromGroup) for n in topNode.contains]))
	num = groupSimilarFromProcess(topNode, num)
	print(num)
	'''
	s1 = set()
	s2 = set()
	for n, node in topNode.contains.items():
		if len(node.toGroup)==0:# and len(node.fromGroup)==1:
			s1.add(node)
		if len(node.fromGroup)==0 and node.type2==2:# and len(node.fromGroup)==1:
			s2.add(node)
		if node.nextVer!=None:
			print(node.name)
	print('len(s1)=',len(s1), 'Edges =', sum([len(node.fromGroup) for node in s1]))
	print('len(s2)=',len(s2), 'Edges =', sum([len(node.toGroup) for node in s2]))
	print('Vertices =', len(topNode.contains), 'Edges =', sum([len(topNode.contains[n].fromGroup) for n in topNode.contains]))
	print(node.name for node in s1)
	'''
	'''
	while True:
		num = groupSimilar(topNode, num)
		print('S\t', num, 'Vertices =', len(topNode.contains), 'Edges =', sum([len(topNode.contains[n].fromGroup) for n in topNode.contains]))
		
		for n in topNode.contains:
			node = topNode.contains[n]
			for n2 in node.fromGroup:
				node2 = node.fromGroup[n2]
				if n not in node2.toGroup:
					print(n,n2)
			for n2 in node.toGroup:
				node2 = node.toGroup[n2]
				if n not in node2.fromGroup:
					print(n,n2)
		#if num==lastnum:
		num = groupPackable_3(topNode, num)
		print('P3\t', num, 'Vertices =', len(topNode.contains), 'Edges =', sum([len(topNode.contains[n].fromGroup) for n in topNode.contains]))
		
		
		if num==lastnum:
			num = groupPackable_2(topNode, num)
			print('P2\t', num, 'Vertices =', len(topNode.contains), 'Edges =', sum([len(topNode.contains[n].fromGroup) for n in topNode.contains]))
		if num==lastnum:
			break
		lastnum = num

	'''

	
	'''

	lastnum = -1
	#num = 0
	while True:
		num = groupSimilar(topNode, num)
		print('S\t', num)
		#num = groupSimilar6(topNode, num)
		#print(num)
		
		#if num==lastnum:
		num = groupPackable_3(topNode, num)
		print('P3\t', num)
		if num==lastnum:
			num = groupPackable_2(topNode, num)
			print('P2\t', num)

		if num==lastnum:
			break
		lastnum = num
	print('Vertices =', len(topNode.contains), 'Edges =', sum([len(topNode.contains[n].fromGroup) for n in topNode.contains]))
	
	
	#Undo unnessesary groupings
	
	ungroup(topNode)
	print('Vertices =', len(topNode.contains))
	'''
	#divide large groups into parts arbitrarily
	divideSimGroupsArbitrarily(topNode)
	print('Vertices =', len(topNode.contains))
	


	'''
	#Debug
	for n in topNode.contains:
		topNode.contains[n].print_()
	
	allNodes['P#10926'].containedIn.print_()
	for n in allNodes['P#10926'].containedIn.toGroup:
		allNodes['P#10926'].containedIn.toGroup[n].print_()
	allNodes['P#10929'].containedIn.print_()
	topNode.contains['P#12048#'].print_()
	'''

	###############<Write to file>##############
	
	writeOutput(allNodes, fileNodes)


if __name__ == "__main__":
	main()
	sys.exit(-1)
else:
	sys.exit(-1)
