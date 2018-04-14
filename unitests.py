'''
To run test files, type the following into the terminal
py.test file.py –vv
'''

import unittest as unittest
from folder.file import MyClass

class TestMyClass(unittest.TestCase):

    def test_myfunction(self):
        myinstance = MyClass(someinput)
        actual_output = myinstance.myfunction()
        correct_output = value
        self.assertAlmostEqual(actual_output, correct_output, \
            msg='The actual output was {}, the correct output is {}.'.format(actual_output,correct_output))

if __name__ == '__main__':
    unittest.main()
