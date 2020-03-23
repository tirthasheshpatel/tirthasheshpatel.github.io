---
type: post
title: Why I hate Python's auto-formating tools like Black
category: Python
tags: [python]
---

![black](https://raw.githubusercontent.com/psf/black/master/docs/_static/logo2-readme.png)

### Why

Supoose you are, like me, working on a scientific Python project and want your code to look beautiful. A natural choice would be using some auto-formatter which will help you follow the same convention throughout the project without worrying about the format while writing the code. While this seems easy, you mostly would end up with a filthy mess that looks more like vomit than "beautiful" code. OTOH, I also believe that following a common convention helps and its almost impossible to keep track of it while writing code. Well, then why do I hate it so much?

Let's say you have a function to calculate the rotation matrix from Euler's angles like follows:

```python
def get_rotation_matrix(rotation_angles):
    sai   = rotation_angles[0] # s
    theta = rotation_angles[1] # t
    phi   = rotation_angles[2] # p
    # find all the required sines and cosines
    cs = np.cos(sai)
    ct = np.cos(theta)
    cp = np.cos(phi)
    ss = np.sin(sai)
    st = np.sin(theta)
    sp = np.sin(phi)
    # contruct the rotation matrix along the x-axis
    rotation_matrix = np.array([
        [ct*cp   ,   ss*st*cp - cs*sp   ,   cs*st*cp + ss*sp],
        [ct*sp   ,   ss*st*sp + cs*cp   ,   cs*st*sp - ss*cp],
        [  -st   ,              sp*ct   ,              cp*ct]
    ])
    return rotation_matrix
```

As you can see, I have tried to align things so it's easier for the reader to go throught the code without staring at it forever. You can clearly see the seperation between the columns of the matrix and make out the equations written in each column and row just by a single look at it. Now, let's format it using a code formatter called $$\mathcal{BLACK}$$.

```python
def get_rotation_matrix(rotation_angles):
    sai = rotation_angles[0]  # s
    theta = rotation_angles[1]  # t
    phi = rotation_angles[2]  # p
    # find all the required sines and cosines
    cs = np.cos(sai)
    ct = np.cos(theta)
    cp = np.cos(phi)
    ss = np.sin(sai)
    st = np.sin(theta)
    sp = np.sin(phi)
    # contruct the rotation matrix along the x-axis
    rotation_matrix = np.array(
        [
            [ct * cp, ss * st * cp - cs * sp, cs * st * cp + ss * sp],
            [ct * sp, ss * st * sp + cs * cp, cs * st * sp - ss * cp],
            [-st, sp * ct, cp * ct],
        ]
    )
    return rotation_matrix
```

This may look visual to the eye but let's try to figure out the equation in the entires of the ``rotation_matrix``. First of all, it is difficult to seperate the columns from each other. Everthing has somehow collapsed onto each other. Other problem is the operations are evenly spaced which makes it harder to differentiate between additions/substractions and multiplications/divisions.

Clearly, the code sample above is **not** beautiful to a scienctific pythoneer's eye.

### What then

For me the best thing is to do is to keep a track of formatting while writting code and not rely entirely on code-formatters like Black. They could be useful to format tests and benchmarks but I would never use them to format the code modules.
