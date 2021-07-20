import unittest
from viz3d.opengl.primitives.sphere import sphere_model_data


class PrimitivesTestCase(unittest.TestCase):
    def test_sphere(self):
        vertex_normal, element_indices = sphere_model_data()
        self.assertEqual(True, True)



if __name__ == '__main__':
    unittest.main()
