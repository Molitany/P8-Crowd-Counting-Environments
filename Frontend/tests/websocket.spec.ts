import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
    await page.goto('localhost:19000');
});

test.describe('testing websocket', () => {
    test('should be able to establish websocket', async ({ page }) => {
        
    });

    test('should be able to display transfered data', async ({ page }) => {
        
    });
});