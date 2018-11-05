object cnn{


def length_list(list:List[Double]):Int = {
  if (list.isEmpty) 0
  else {
    1 + length_list(list.tail)
  }
}

def length_matrix(matrix:List[List[Double]]):Int = {
  if (matrix.isEmpty) 0
  else {
    1 + length_matrix(matrix.tail)
  }
}

def listmul(list_1:List[Double], list_2:List[Double]):List[Double] = {
  if ( list_1.isEmpty ) Nil
  else {
    List(list_1.head * list_2.head) ::: listmul( list_1.tail, list_2.tail)
  }
}

def listadd(list_1:List[Double], list_2:List[Double]):List[Double] = {
  if ( list_1.isEmpty ) Nil
  else {
    List(list_1.head + list_2.head) ::: listadd( list_1.tail, list_2.tail)
  }
}

def matrixadd(matrix_1:List[List[Double]], matrix_2:List[List[Double]]):List[List[Double]] = {
  if ( matrix_1.isEmpty ) Nil
  else {
    List(listadd(matrix_1.head, matrix_2.head)) ::: matrixadd(matrix_1.tail, matrix_2.tail)
  }
}

def listsum(list:List[Double]):Double = {
  if (list.isEmpty) 0
  else{
    list.head + listsum(list.tail)
  }
}

def listdot(list_1:List[Double], list_2:List[Double]):Double = {
  listsum(listmul(list_1, list_2))
}

//------------------ 1.1 Dot Product

def dotProduct(matrix_1:List[List[Double]], matrix_2:List[List[Double]]):Double = {
  //matrix_1 is kernel
  if ( matrix_1.isEmpty ) 0
  else {
    listdot(matrix_1.head, matrix_2.head) + dotProduct(matrix_1.tail, matrix_2.tail)
  }
}


def delete_first_column(matrix:List[List[Double]]):List[List[Double]] = {
  if (matrix.isEmpty) Nil
  else {
    List(matrix.head.tail) ::: delete_first_column(matrix.tail)
  }
}

def retain_first_column(matrix:List[List[Double]]):List[Double] = {
  if (matrix.isEmpty) Nil
  else {
    List(matrix.head.head) ::: retain_first_column(matrix.tail)
  }
}

def decrement_columns(imageSize:List[Int]):List[Int] = {
  List(imageSize.head) ::: List(imageSize.tail.head - 1)
}

def convolute_mini(Image:List[List[Double]], Kernel:List[List[Double]], imageSize:List[Int], kernelSize:List[Int]):List[Double] = {
  if (imageSize.tail.head < kernelSize.tail.head) Nil
  else {
    List(dotProduct(Kernel, Image)) ::: convolute_mini(delete_first_column(Image), Kernel, decrement_columns(imageSize), kernelSize)
  }
}

// -------------------------1.2 Convolute
def convolute(Image:List[List[Double]], Kernel:List[List[Double]], imageSize:List[Int], kernelSize:List[Int]):List[List[Double]] = {
    if (imageSize.head < kernelSize.head) Nil
    else {
      List(convolute_mini(Image, Kernel, imageSize, kernelSize)) ::: convolute(Image.tail, Kernel, List(imageSize.head-1):::imageSize.tail, kernelSize)
    }
}

// def activationFunc(x:Double):Double = {x-1}
def ReLu(x:Double):Double = {
  if (x > 0) {x}
  else {0}
}
def leaky_ReLu(x:Double):Double = {
  if (x > 0) {x}
  else {0.5*x}
}

def multiply(x:Double, w:Double):Double = {x*w}
def add(x:Double, b:Double):Double = {x+b}

def activationLayer_mini(activationFunc:Double => Double, Image_mini:List[Double]):List[Double] = {
  if (Image_mini.isEmpty) Nil
  else {
    List(activationFunc(Image_mini.head)) ::: activationLayer_mini(activationFunc, Image_mini.tail)
  }
}

// -------------------------2 Activation Layer
def activationLayer(activationFunc:Double => Double, Image:List[List[Double]]):List[List[Double]] = {
  if (Image.isEmpty) Nil
  else {
    List(activationLayer_mini(activationFunc, Image.head)) ::: activationLayer(activationFunc, Image.tail)
  }
}

def weightmultiply_mini(Image_mini:List[Double], w:Double):List[Double] = {
  if (Image_mini.isEmpty) Nil
  else {
    List(multiply(Image_mini.head, w)) ::: weightmultiply_mini(Image_mini.tail, w)
  }
}

def weightmultiply(Image:List[List[Double]], w:Double):List[List[Double]] = {
  if (Image.isEmpty) Nil
  else {
    List(weightmultiply_mini(Image.head, w)) ::: weightmultiply(Image.tail, w)
  }
}

def biasadd_mini(Image_mini:List[Double], b:Double):List[Double] = {
  if (Image_mini.isEmpty) Nil
  else {
    List(add(Image_mini.head, b)) ::: biasadd_mini(Image_mini.tail, b)
  }
}

def biasadd(Image:List[List[Double]], b:Double):List[List[Double]] = {
  if (Image.isEmpty) Nil
  else {
    List(biasadd_mini(Image.head, b)) ::: biasadd(Image.tail, b)
  }
}

def maxFunc(list:List[Double]):Double = list match {
    case Nil => Double.MinValue
    case a :: Nil => a
    case a :: b :: c =>  max( (if (a > b) a else b) ::c)
  }

def max(list:List[Double]):Double = list match {
    case Nil => Double.MinValue
    case a :: Nil => a
    case a :: b :: c =>  max( (if (a > b) a else b) ::c)
  }

def avg(list:List[Double]):Double = {
    listsum(list)/length_list(list)
}

def minFunc(list:List[Double]):Double = list match {
    case Nil => Double.MaxValue
    case a :: Nil => a
    case a :: b :: c =>  minFunc( (if (a > b) b else a) ::c)
  }


def twoD_oneD(list:List[List[Double]], k:Int):List[Double] = {
  if (k == 0) Nil
  else {
    retain_first_column(list) ::: twoD_oneD(delete_first_column(list), k-1)
  }
}

def delete_k_column(matrix:List[List[Double]], k:Int):List[List[Double]] = {
  if (k==0) matrix
  else {
    delete_k_column(delete_first_column(matrix), k-1)
  }
}

// -----------------3.1 Single Pooling
def singlePooling(poolingFunc:List[Double]=>Double, Image:List[List[Double]], K:Int):List[Double] = {
    if (length_list(Image.head) == K) List(poolingFunc(twoD_oneD(Image, K)))
    else {
      List(poolingFunc(twoD_oneD(Image, K))) ::: singlePooling(poolingFunc, delete_k_column(Image, K), K)
    }
}

val image_pool:List[List[Double]] = List(List(25,40,55,70), List(40,64,88,112), List(55,88,121,154), List(70,112,154,196))

def first_k_rows(matrix:List[List[Double]], k:Int):List[List[Double]] = {
  if (k <= 0) Nil
  else {
    List(matrix.head) ::: first_k_rows(matrix.tail, k-1)
  }
}

def delete_k_rows(matrix:List[List[Double]], k:Int):List[List[Double]] = {
  if (k <= 0) matrix
  else {
    delete_k_rows(matrix.tail, k-1)
  }
}

// -------------------3.2 Pooling Layer
def poolingLayer(poolingFunc:List[Double]=>Double, Image:List[List[Double]], K:Int):List[List[Double]] = {
  if(length_matrix(Image) == K) List(singlePooling(poolingFunc, Image, K))
  else {
    List(singlePooling(poolingFunc, first_k_rows(Image, K), K)) ::: poolingLayer(poolingFunc, delete_k_rows(Image, K), K)
  }
}

// ---------------------4 Normalization
def mixedLayer(Image:List[List[Double]],Kernel:List[List[Double]], imageSize:List[Int], kernelSize:List[Int], activationFunc:Double => Double, poolingFunc:List[Double]=>Double, K:Int):List[List[Double]] = {
    poolingLayer(poolingFunc,activationLayer(activationFunc, convolute(Image, Kernel, imageSize, kernelSize)), K)
}

def matrix_max(matrix:List[List[Double]]):Double = {
  maxFunc(twoD_oneD(matrix, length_matrix(matrix)))
}

def matrix_min(matrix:List[List[Double]]):Double = {
  minFunc(twoD_oneD(matrix, length_matrix(matrix)))
}

def normalise_mini(list:List[Double], maxa:Double, min:Double):List[Int] = {
  if (list.isEmpty) Nil
  else {
    List((((list.head - min)/(maxa-min))*255.0).round.toInt) ::: normalise_mini(list.tail, maxa, min)
  }
}

def preserve_max_min(Original:List[List[Double]], Image:List[List[Double]]):List[List[Int]] = {
  if (Image.isEmpty) Nil
  else {
    List(normalise_mini(Image.head, matrix_max(Original), matrix_min(Original))) ::: preserve_max_min(Original, Image.tail)
  }
}

// ----------------------5 Mixed layer
def normalise(Image:List[List[Double]]):List[List[Int]] = {
  preserve_max_min(Image, Image)
}

def find_size(matrix:List[List[Double]]):List[Int] = {
  List(length_matrix(matrix),length_list(matrix.head))
}

// -----------------------6 Assembly
def assembly(Image:List[List[Double]], imageSize:List[Int], w1:Double, w2:Double, b:Double, Kernel1:List[List[Double]], kernelSize1:List[Int], Kernel2:List[List[Double]], kernelSize2:List[Int], Kernel3:List[List[Double]], kernelSize3:List[Int], Size: Int):List[List[Int]] = {
/*
val tempout1 = mixedLayer(Image, Kernel1, imageSize, kernelSize1, ReLu, avg, Size)
val tempout2 = mixedLayer(Image, Kernel2, imageSize, kernelSize2, ReLu, avg, Size)

val tempout3 = biasadd(matrixadd(weightmultiply(tempout1, w1), weightmultiply(tempout2, w2)), b)

val tempout4 = mixedLayer(tempout3, Kernel3, find_size(tempout3), kernelSize3, leaky_ReLu, max, Size)

normalise(tempout4)
*/

normalise(mixedLayer(biasadd(matrixadd(weightmultiply(mixedLayer(Image, Kernel1, imageSize, kernelSize1, ReLu, avg, Size), w1), weightmultiply(mixedLayer(Image, Kernel2, imageSize, kernelSize2, ReLu, avg, Size), w2)), b), Kernel3, find_size(mixedLayer(Image, Kernel1, imageSize, kernelSize1, ReLu, avg, Size)), kernelSize3, leaky_ReLu, max, Size))

}
}
