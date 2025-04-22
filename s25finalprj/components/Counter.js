"use client";

import { useState } from "react";

function Counter() {
    const [count, setCount] = useState(0);
    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={() => setCount((prevCount) => prevCount + 1)}>Increment</button>
            <button onClick={() => setCount((prevCount) => prevCount - 1)}>Decrement</button>
        </div>
    );
}

export default Counter;


