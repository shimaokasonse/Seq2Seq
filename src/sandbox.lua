local BatchLoader = require 'BatchLoader'

loader = BatchLoader.create("../data/dataset.t7",2,2)

for i=1,100 do
  x,y = loader:next_batch()
  print("x:")
  print(x)
  print("y:")
  print(y)
  print("\n")
end
